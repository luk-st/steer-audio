import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import rootutils
import scipy.io.wavfile as wav
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import torchaudio
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
from src.metrics import calculate_clap, calculate_fad, calculate_music_alignment, calculate_muqt
from src.utils import RankedLogger, extras, save_dict_to_json, task_wrapper
from src.eval_audio import load_audios_save, save_per_prompt_sims_to_csv, del_audios_dir
from src.patch_layers import extract_prompts_csv

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def eval_prompts(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    assert cfg.patch_config

    metrics = {}

    generated_samples_path, reference_samples_path = None, None
    generated_samples_path = Path(os.path.join(cfg.paths.all_output_dir, cfg.paths.generated_samples))
    generated_samples_path = load_audios_save(generated_samples_path, cfg.patch_model.config.sample_rate)

    path_to_save_metrics = (Path(cfg.paths.all_output_dir) / Path(cfg.paths.generated_samples).parent.parent).resolve()

    # if cfg.metrics.fad or cfg.metrics.alignment:
    #     reference_samples_path = Path(os.path.join(cfg.paths.all_output_dir, cfg.paths.reference_samples))
    #     reference_samples_path = load_audios_save(reference_samples_path, cfg.patch_model.config.sample_rate)
    
    if cfg.patch_data.type == "csv":
        clean_prompts, corrupted_prompts = extract_prompts_csv(
            csv_path=cfg.patch_data.csv_path,
            feature=cfg.patch_data.feature,
            prefix_clean=cfg.patch_data.prefix_clean,
            suffix_clean=cfg.patch_data.suffix_clean,
            prefix_corrupted=cfg.patch_data.prefix_corrupted,
            suffix_corrupted=cfg.patch_data.suffix_corrupted,
        )
        if cfg.patch_config.method == "ablate":
            corrupted_prompts = None
    if cfg.patch_config.n_latents_per_prompt and cfg.patch_config.n_latents_per_prompt > 1:
        clean_prompts = [p for p in clean_prompts for _ in range(cfg.patch_config.n_latents_per_prompt)]
        corrupted_prompts = [p for p in corrupted_prompts for _ in range(cfg.patch_config.n_latents_per_prompt)] if corrupted_prompts else None

    elif cfg.patch_data.type == "list":
        clean_prompts = cfg.patch_data.prompts
        corrupted_prompts = cfg.patch_data.prompts_patch if cfg.patch_config.method == "patch" else None

    if cfg.patch_config.n_latents_per_prompt and cfg.patch_config.n_latents_per_prompt > 1:
        clean_prompts = [p for p in clean_prompts for _ in range(cfg.patch_config.n_latents_per_prompt)]
        corrupted_prompts = [p for p in corrupted_prompts for _ in range(cfg.patch_config.n_latents_per_prompt)] if corrupted_prompts else None

    log.info("Calculating metrics...")
    if cfg.metrics.fad:
        log.info("Calculating FAD...")
        metrics["fad"] = calculate_fad(
            generated_samples_path=generated_samples_path,
            reference_samples_path=reference_samples_path,
        )

    if cfg.metrics.alignment:
        log.info("Calculating music alignment...")
        metrics["alignment"] = calculate_music_alignment(
            generated_samples_path=generated_samples_path,
            reference_samples_path=reference_samples_path,
            sampling_rate=cfg.patch_model.config.sample_rate,
            device=cfg.device,
        )

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
            save_per_prompt_sims_to_csv(per_prompt_sims_clap, path_to_save_metrics, "clap")

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
            save_per_prompt_sims_to_csv(per_prompt_sims_clap_music, path_to_save_metrics, "clap_music")

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
            save_per_prompt_sims_to_csv(per_prompt_sims_muqt, path_to_save_metrics, "muqt")

    for key, value in metrics.items():
        log.info(f"{key}: {value}")

    object_dict = {
        "cfg": cfg,
    }

    out_json_path = (path_to_save_metrics / "metrics.json")
    log.info(f"Metrics saved to {out_json_path}")
    save_dict_to_json(
        metrics, out_json_path
    )

    if cfg.keep_audios is False:
        del_audios_dir(generated_samples_path)
        # if reference_samples_path:
        #     del_audios_dir(reference_samples_path)

    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_audio_prompts.yaml")
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
