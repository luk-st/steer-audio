"""Audio editing / steering evaluation metrics.

Three windowed metrics used throughout the project's steering eval pipeline:

* :func:`get_lpaps`   — perceptual audio distance (LPAPS) between source and edited audio.
* :func:`get_clap`    — text-audio alignment (CLAP) between edited audio and target prompt.
* :func:`get_mulan`   — MuQ-MuLan text-audio similarity, an independent second alignment metric.

All three load their backbone model on call, run windowed evaluation, and return a
pandas DataFrame keyed by audio index. The CLAP backbone is the music-finetuned
LAION-CLAP checkpoint expected at ``res/clap/pretrained/music_audioset_epoch_15_esc_90.14.pt``
(see the README's "Model checkpoints" section).

Imported by:
- ``src/steering/eval/eval_steering_protocol.py``        (get_lpaps + get_clap)
- ``src/steering/methods/caa/generate_baseline_variance.py``  (get_lpaps)
- ``src/steering/methods/caa/utils/eval_utils.py``      (get_mulan)

The legacy medley-editing CLI / ZoME benchmark code lives under ``editing/legacy/``;
nothing in the active steering pipeline imports from there.
"""

from __future__ import annotations

import os

import pandas as pd
import torch
from muq import MuQMuLan
from torchaudio.transforms import Resample
from tqdm import tqdm

from editing.AudioEditingCode.evals.lpaps import LPAPS
from editing.AudioEditingCode.evals.meta_clap_consistency import (
    CLAPTextConsistencyMetric,
)
from editing.AudioEditingCode.evals.utils import calc_clap_win, calc_lpaps_win

DISABLE_TQDM = False
_CLAP_CKPT = "music_audioset_epoch_15_esc_90.14.pt"
_CLAP_DIR = "res/clap/pretrained"
_CLAP_PATH = os.path.join(_CLAP_DIR, _CLAP_CKPT)
_FUSION = "fusion" in _CLAP_CKPT
_MODEL_ARCH = "HTSAT-tiny" if _FUSION else "HTSAT-base"
_WIN_LENGTH = None if _FUSION else 10


def get_lpaps(source_audios, edits, srs_src, srs_edit, device) -> pd.DataFrame:
    """Per-pair LPAPS (perceptual audio similarity) between source and edited audio.

    Returns a DataFrame with columns ``audio_idx``, ``lpaps``.
    """
    lpaps_model = LPAPS(
        net="clap",
        device=device,
        net_kwargs={
            "model_arch": _MODEL_ARCH,
            "chkpt": _CLAP_CKPT,
            "enable_fusion": _FUSION,
        },
        checkpoint_path=_CLAP_DIR,
    )
    scores = {}
    with torch.no_grad():
        for i in tqdm(range(len(source_audios)), desc="LPAPS", disable=DISABLE_TQDM):
            scores[i] = calc_lpaps_win(
                lpaps_model=lpaps_model,
                aud1=source_audios[i],
                aud2=edits[i],
                sr1=srs_src[i],
                sr2=srs_edit[i],
                win_length=_WIN_LENGTH,
                overlap=0.1,
                method="mean",
                device=device,
            )
    return pd.DataFrame(list(scores.items()), columns=["audio_idx", "lpaps"])


def get_clap(target_prompts, edits, srs_edit, device) -> pd.DataFrame:
    """Per-pair CLAP text-audio similarity to ``target_prompts``.

    Returns a DataFrame indexed by audio_idx with columns ``clap`` and ``prompt``.
    """
    clap_model = (
        CLAPTextConsistencyMetric(
            model_path=_CLAP_PATH,
            model_arch=_MODEL_ARCH,
            enable_fusion=_FUSION,
        )
        .to(device)
        .eval()
    )

    scores = {}
    with torch.no_grad():
        for i in tqdm(range(len(edits)), desc="CLAP", disable=DISABLE_TQDM):
            scores[i] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=edits[i],
                    sr=srs_edit[i],
                    target_prompt=target_prompts[i],
                    win_length=_WIN_LENGTH,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": target_prompts[i],
            }
    return pd.DataFrame.from_dict(scores, orient="index")


def get_mulan(
    target_prompts, edits, srs_edit, device, verbose: bool = True
) -> pd.DataFrame:
    """Per-pair MuQ-MuLan text-audio similarity to ``target_prompts``.

    Returns a DataFrame indexed by audio_idx with columns ``muqt_sim_p0`` and ``p0``.
    """
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()

    sims: list = []
    with torch.no_grad():
        for i in tqdm(
            range(len(edits)), desc="MUQT", disable=DISABLE_TQDM or not verbose
        ):
            text_embeds = mulan(texts=[target_prompts[i]])
            wav = Resample(srs_edit[i], 24000)(edits[i]).squeeze(1).to(mulan.device)
            audio_embeds = mulan(wavs=wav)
            sims.append(mulan.calc_similarity(audio_embeds, text_embeds).cpu())
        similarities = torch.cat(sims, dim=0)

    out = {
        i: {
            "muqt_sim_p0": similarities[i, 0].item(),
            "p0": target_prompts[i],
        }
        for i in range(len(edits))
    }
    return pd.DataFrame.from_dict(out, orient="index")
