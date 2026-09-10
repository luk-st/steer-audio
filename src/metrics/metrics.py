from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

CLAP_MODEL = "laion/clap-htsat-unfused"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_clap(
    audio_dir: str,
    prompts: List[str],
    clap_prompt_template: str = "{p}",
    use_music_checkpoint: bool = False,
    device: str = "cuda",
    batch_size: int = 64,
):
    import laion_clap

    # PyTorch 2.6+ torch.load's fix by monkey patch
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})  # type: ignore
    try:
        if use_music_checkpoint:
            clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            clap_model.load_ckpt(
                "res/clap/pretrained/music_audioset_epoch_15_esc_90.14.pt",
                verbose=False,
            )
        else:
            clap_model = laion_clap.CLAP_Module(enable_fusion=True)
            clap_model.load_ckpt(verbose=False)
    finally:
        torch.load = _orig_torch_load
    clap_model = clap_model.to(torch.device(device))
    clap_model = clap_model.eval()

    from src.steering.eval.audio_io import as_wav_dir

    with as_wav_dir(audio_dir) as _wd:  # packed audios.npz -> temp wavs; else the dir itself
        audio_files = sorted(Path(_wd).glob("*.wav"))
        audio_files = [str(p) for p in audio_files]

        with torch.no_grad():
            all_texts = [clap_prompt_template.format(p=prompt) for prompt in prompts]
            text_embed = torch.tensor(clap_model.get_text_embedding(all_texts)).cpu()
            all_audio_embeds = []
            for i in range(0, len(audio_files), batch_size):
                batch_files = audio_files[i : i + batch_size]
                batch_embed = clap_model.get_audio_embedding_from_filelist(x=batch_files)
                all_audio_embeds.append(torch.tensor(batch_embed).cpu())

            audio_embed = torch.cat(all_audio_embeds, dim=0)

            # Normalize embeddings to unit length for cosine similarity
            audio_embed = torch.nn.functional.normalize(audio_embed, p=2, dim=1)
            text_embed = torch.nn.functional.normalize(text_embed, p=2, dim=1)

            sims = audio_embed @ text_embed.t()
            means = sims.mean(dim=0).float()
            stds = sims.std(dim=0).float()

    per_prompt_sims = {}
    model_name = "clapmusic" if use_music_checkpoint else "clap"
    for audio_idx in range(len(audio_files)):
        per_prompt_sims[audio_idx] = {}
        for p_idx in range(len(prompts)):
            per_prompt_sims[audio_idx][f"{model_name}_sim_p{p_idx}"] = sims[audio_idx, p_idx].item()
            per_prompt_sims[audio_idx][f"p{p_idx}"] = prompts[p_idx]

    return {
        text: {
            "mean": f"{means[i]:.3f}",
            "std": f"{stds[i]:.3f}",
        }
        for i, text in enumerate(prompts)
    }, per_prompt_sims


def calculate_muqt(
    audio_dir: str,
    prompts: List[str],
    prompt_template: str = "{p}",
    device: str = "cuda",
    batch_size: int = 64,
    sr: int = 44100,
    resample_to_24k: bool = False,
):
    import librosa
    from muq import MuQMuLan

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    from src.steering.eval.audio_io import as_wav_dir

    with torch.no_grad():
        all_texts = [prompt_template.format(p=prompt) for prompt in prompts]
        text_embeds = mulan(texts=all_texts)

    with as_wav_dir(audio_dir) as _wd:  # packed audios.npz -> temp wavs; else the dir itself
        audio_files = sorted(Path(_wd).glob("*.wav"))
        audio_files = [str(p) for p in audio_files]
        with torch.no_grad():
            all_similarities = []
            for i in tqdm(range(0, len(audio_files), batch_size), desc="Calculating MuQT"):
                batch_files = audio_files[i : i + batch_size]
                batch_files = [librosa.load(file, sr=sr)[0] for file in batch_files]
                if resample_to_24k:
                    batch_files = [librosa.resample(file, orig_sr=sr, target_sr=24000) for file in batch_files]
                batch_wavs_np = np.array(batch_files)
                batch_wavs = torch.from_numpy(batch_wavs_np).to(device)
                batch_embed = mulan(wavs=batch_wavs)
                batch_similarities = mulan.calc_similarity(batch_embed, text_embeds)
                all_similarities.append(batch_similarities.cpu())
            similarities = torch.cat(all_similarities, dim=0)

    mean_pp = similarities.mean(dim=0).float()
    std_pp = similarities.std(dim=0).float()

    per_prompt_sims = {}
    for audio_idx in range(len(audio_files)):
        per_prompt_sims[audio_idx] = {}
        for p_idx in range(len(prompts)):
            per_prompt_sims[audio_idx][f"muqt_sim_p{p_idx}"] = similarities[audio_idx, p_idx].item()
            per_prompt_sims[audio_idx][f"p{p_idx}"] = prompts[p_idx]

    return {
        text: {
            "mean": f"{mean_pp[i]:.3f}",
            "std": f"{std_pp[i]:.3f}",
        }
        for i, text in enumerate(prompts)
    }, per_prompt_sims
