import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import rootutils
import scipy.io.wavfile as wav
import soundfile as sf
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import pandas as pd

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.metrics import (
    calculate_clap,
    calculate_muqt,
)
from src.utils import RankedLogger, extras, save_dict_to_json, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def save_per_prompt_sims_to_csv(
    per_prompt_sims: Dict[int, Dict[str, float]], output_dir: Path, metric_name: str
) -> None:
    """Saves per-prompt similarities to a CSV file."""
    df = pd.DataFrame.from_dict(per_prompt_sims, orient="index")
    df.to_csv(output_dir / f"pp_sims_{metric_name}.csv")


def load_audios_save(path: Path, sample_rate: int) -> Path:
    """Loads audios from a numpy file and saves them as wav files."""
    assert path.suffix == ".npy"
    file_name = path.stem
    dir_path = (path.parent / f"audios_{file_name}").resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Loading audios from {path}")

    if not path.exists():
        log.info(
            f"Path {path} does not exist, searching for files with name {file_name}_*.npy"
        )
        # find all .npy files that are named {file_name}_{int}.npy
        npy_files = list(path.parent.glob(file_name + "_*.npy"))
        npy_files = sorted(npy_files, key=lambda x: int(x.stem.split("_")[-1]))
        log.info(f"Found {len(npy_files)} files")
    else:
        npy_files = [path]

    idx = 0
    for npy_path in npy_files:
        audios = np.load(npy_path)
        audios_tensor = torch.from_numpy(audios)
        if len(audios_tensor.shape) == 2:
            # add channel dim
            audios_tensor = audios_tensor.unsqueeze(1)

        for _, audio in tqdm(
            enumerate(audios_tensor),
            total=len(audios_tensor),
            desc=f"Saving audios: {dir_path}",
        ):
            # torchaudio>=2.9 routes save() through torchcodec, which dlopens
            # FFmpeg (libavutil.so.56-59); WCSS nodes have none. libsndfile needs
            # no FFmpeg, and round(x*32768) with clipping reproduces torchcodec's
            # PCM_16 quantization, so samples stay bit-identical to earlier runs.
            samples = (
                (audio.float() * 32768.0).round().clamp(-32768.0, 32767.0).to(torch.int16)
            )
            sf.write(
                str((dir_path / f"a_id{idx}.wav").resolve()),
                samples.numpy().T,
                sample_rate,
                subtype="PCM_16",
            )
            idx += 1

    log.info(f"Saving N={idx} audios to {dir_path}")

    return dir_path


def del_audios_dir(path: Path) -> None:
    """Deletes the audios directory."""
    log.info(f"Deleting audios directory: {path}")
    shutil.rmtree(path)


@task_wrapper
def eval(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.metrics
    assert cfg.paths
    assert cfg.patch_data
    assert cfg.device
    assert cfg.patch_model
    assert cfg.clap_prompt_template

    metrics = {}

    generated_samples_path, reference_samples_path = None, None
    generated_samples_path = Path(
        os.path.join(cfg.paths.all_output_dir, cfg.paths.generated_samples)
    )
    generated_samples_path = load_audios_save(
        generated_samples_path, cfg.patch_model.config.sample_rate
    )

    path_to_save_metrics = (
        Path(cfg.paths.all_output_dir) / Path(cfg.paths.generated_samples).parent.parent
    ).resolve()

    if cfg.metrics.clap:
        log.info("Calculating CLAP (basic)...")
        metrics["clap"], per_prompt_sims_clap = calculate_clap(
            audio_dir=generated_samples_path,
            prompts=list(cfg.patch_data.eval_clap_prompts),
            clap_prompt_template=cfg.clap_prompt_template,
            use_music_checkpoint=False,
            batch_size=cfg.clap_batch_size,
            device=cfg.device,
        )
        if cfg.save_per_prompt_sims:
            save_per_prompt_sims_to_csv(
                per_prompt_sims_clap, path_to_save_metrics, "clap"
            )

    if cfg.metrics.clap_music:
        log.info("Calculating CLAP (music)...")
        metrics["clap_music"], per_prompt_sims_clap_music = calculate_clap(
            audio_dir=generated_samples_path,
            prompts=list(cfg.patch_data.eval_clap_prompts),
            clap_prompt_template=cfg.clap_prompt_template,
            use_music_checkpoint=True,
            batch_size=cfg.clap_batch_size,
            device=cfg.device,
        )
        if cfg.save_per_prompt_sims:
            save_per_prompt_sims_to_csv(
                per_prompt_sims_clap_music, path_to_save_metrics, "clap_music"
            )

    if cfg.metrics.muqt:
        log.info("Calculating MuQT...")
        metrics["muqt"], per_prompt_sims_muqt = calculate_muqt(
            audio_dir=generated_samples_path,
            prompts=list(cfg.patch_data.eval_muqt_prompts),
            prompt_template=cfg.muqt_prompt_template,
            batch_size=cfg.muqt_batch_size,
            device=cfg.device,
            sr=cfg.patch_model.config.sample_rate,
            resample_to_24k=cfg.muqt_resample_to_24k,
        )
        if cfg.save_per_prompt_sims:
            save_per_prompt_sims_to_csv(
                per_prompt_sims_muqt, path_to_save_metrics, "muqt"
            )

    for key, value in metrics.items():
        log.info(f"{key}: {value}")

    object_dict = {
        "cfg": cfg,
    }

    out_json_path = path_to_save_metrics / "metrics.json"
    log.info(f"Metrics saved to {out_json_path}")
    save_dict_to_json(metrics, out_json_path)

    if cfg.keep_audios is False:
        del_audios_dir(generated_samples_path)
        if reference_samples_path:
            del_audios_dir(reference_samples_path)

    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_audio.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    eval(cfg)


if __name__ == "__main__":
    main()
