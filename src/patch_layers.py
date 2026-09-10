from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import rootutils
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocess.utils import extract_prompts_csv
from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def generate_with_patch(cfg: DictConfig, accelerator: Accelerator) -> Dict[str, Any]:
    """Evaluates given checkpoint on a datamodule testset.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: dict with all instantiated objects.
    """
    assert cfg.patch_model
    assert cfg.patch_data
    assert cfg.patch_config
    assert cfg.patch_layers
    assert cfg.patch_model.model_name in [
        "audioldm2",
        "stableaudio",
        "ace_step",
    ]
    assert cfg.patch_config.method in ["patch", "ablate"]

    if accelerator.is_main_process:
        log.info("Instantiating model...")

    device = accelerator.device
    model_name = cfg.patch_model.model_name
    if model_name in ["audioldm2", "stableaudio"]:
        dtype = getattr(torch, cfg.patch_model.torch_dtype)
        pipeline = hydra.utils.instantiate(
            cfg.patch_model.pipeline,
            dtype=dtype,
            device=device,
            negative_prompt=cfg.patch_config.negative_prompt,
        )
    elif model_name == "ace_step":
        pipeline = hydra.utils.instantiate(
            cfg.patch_model.pipeline,
            device=device,
            pad_to_max_len=cfg.patch_config.pad_to_max_len,
        )

    if accelerator.is_main_process:
        log.info("Generating patched audios...")

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

    elif cfg.patch_data.type == "list":
        clean_prompts = cfg.patch_data.prompts
        corrupted_prompts = (
            cfg.patch_data.prompts_patch if cfg.patch_config.method == "patch" else None
        )

    prompt_limit = cfg.get("prompt_limit")
    if prompt_limit is not None:
        clean_prompts = list(clean_prompts)[:prompt_limit]
        if corrupted_prompts is not None:
            corrupted_prompts = list(corrupted_prompts)[:prompt_limit]
        if accelerator.is_main_process:
            log.info(
                f"Capping to {len(clean_prompts)} prompt pair(s) via prompt_limit={prompt_limit}."
            )

    if (
        cfg.patch_config.n_latents_per_prompt
        and cfg.patch_config.n_latents_per_prompt > 1
    ):
        clean_prompts = [
            p
            for p in clean_prompts
            for _ in range(cfg.patch_config.n_latents_per_prompt)
        ]
        corrupted_prompts = (
            [
                p
                for p in corrupted_prompts
                for _ in range(cfg.patch_config.n_latents_per_prompt)
            ]
            if corrupted_prompts
            else None
        )

    if model_name in ["audioldm2", "stableaudio"]:
        latents = pipeline.prepare_latents(
            n_prompts=len(clean_prompts),
            seed=cfg.patch_config.seed,
            audio_length_in_s=cfg.patch_config.audio_length_in_s,
        )
    elif model_name == "ace_step":
        latents = pipeline.prepare_latents(
            n_prompts=len(clean_prompts),
            seed=cfg.patch_config.seed,
            audio_duration=cfg.patch_config.audio_duration,
        )

    if model_name in ["audioldm2", "stableaudio"]:
        kwargs_sampling = {
            "audio_length_in_s": cfg.patch_config.audio_length_in_s,
        }
        if cfg.patch_config.method == "ablate":
            kwargs_sampling["ablate_null_pred"] = cfg.patch_config.ablate_null_pred
    elif model_name == "ace_step":
        kwargs_sampling = {
            "audio_duration": cfg.patch_config.audio_duration,
            "lyrics": cfg.patch_config.lyrics,
            "guidance_scale_lyric": cfg.patch_config.guidance_scale_lyric,
            "guidance_scale_text": cfg.patch_config.guidance_scale_text,
            "guidance_interval": cfg.patch_config.guidance_interval,
            "scheduler_type": cfg.patch_config.scheduler_type,
            "cfg_type": cfg.patch_config.cfg_type,
            "omega_scale": cfg.patch_config.omega_scale,
            "seed": cfg.patch_config.seed,
        }

    accelerator.wait_for_everyone()
    idxs_prompts = list(range(len(clean_prompts)))
    all_outputs = defaultdict(list)
    if accelerator.is_main_process:
        log.info(
            f"Multi-GPU:\n - Enabled: {accelerator.num_processes > 1}\n - Examples to process: {len(idxs_prompts)}\n - Number of GPUs: {accelerator.num_processes}\n - Examples per GPU: {len(idxs_prompts) // accelerator.num_processes}"
        )
    with accelerator.split_between_processes(idxs_prompts) as device_idxs:
        clean_prompts_device = [clean_prompts[i] for i in device_idxs]
        corrupted_prompts_device = (
            [corrupted_prompts[i] for i in device_idxs] if corrupted_prompts else None
        )
        latents_device = torch.stack([latents[i] for i in device_idxs])

        if cfg.patch_config.method == "patch":
            audios = pipeline.generate_by_patching(
                prompts_clean=clean_prompts_device,
                prompts_corrupted=corrupted_prompts_device,
                layers_to_patch=cfg.patch_layers.layers_to_patch,
                latents=latents_device,
                batch_size=cfg.patch_config.batch_size,
                accelerator=accelerator,
                num_inference_steps=cfg.patch_config.num_inference_steps,
                guidance_scale=cfg.patch_config.guidance_scale,
                **kwargs_sampling,
            )
        elif cfg.patch_config.method == "ablate":
            audios = pipeline.generate_by_ablating(
                prompts_clean=clean_prompts_device,
                layers_to_ablate=cfg.patch_layers.layers_to_patch,
                latents=latents_device,
                batch_size=cfg.patch_config.batch_size,
                accelerator=accelerator,
                num_inference_steps=cfg.patch_config.num_inference_steps,
                guidance_scale=cfg.patch_config.guidance_scale,
                **kwargs_sampling,
            )
        # Optionally downmix to mono to halve on-disk size; the metric encoders
        # (MuQ/CLAP) downmix internally, so this is lossless for the eval.
        save_mono = cfg.patch_config.get("save_mono", False)
        for output_name, output_audios in audios.items():
            if save_mono and output_audios.ndim == 3 and output_audios.shape[1] > 1:
                output_audios = output_audios.mean(axis=1)
            all_outputs[output_name].append(output_audios)
    accelerator.wait_for_everyone()

    if cfg.patch_config.save_per_gpu:
        if cfg.patch_config.save_clean_audio is False:
            if "clean" in all_outputs.keys():
                del all_outputs["clean"]

        for output_name, output_audios in all_outputs.items():
            output_audios = np.concatenate(output_audios, axis=0)
            rank = accelerator.process_index

            if cfg.patch_config.path_with_results is not None:
                out_path = (
                    Path(cfg.patch_config.path_with_results)
                    / cfg.patch_config.audio_dirname
                ).resolve()
            else:
                out_path = (
                    Path(cfg.paths.output_dir) / cfg.patch_config.audio_dirname
                ).resolve()
            audio_out_path = (out_path / f"{output_name}_{rank}.npy").resolve()

            audio_out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(audio_out_path, output_audios)
            log.info(f"Results saved to {audio_out_path}")
    else:
        for obj_name, obj_value in all_outputs.items():
            vals = gather_object(obj_value)
            all_outputs[obj_name] = np.concatenate(vals, axis=0)

        if accelerator.is_main_process:
            log.info("Saving audio results...")

            if cfg.patch_config.path_with_results is not None:
                out_path = (
                    Path(cfg.patch_config.path_with_results)
                    / cfg.patch_config.audio_dirname
                ).resolve()
            else:
                out_path = (
                    Path(cfg.paths.output_dir) / cfg.patch_config.audio_dirname
                ).resolve()

            for output_name, output_audios in all_outputs.items():
                audio_out_path = (out_path / f"{output_name}.npy").resolve()
                audio_out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(audio_out_path, output_audios)

            log.info(f"Results saved to {out_path}")

    object_dict = {"cfg": cfg, "model": pipeline.pipeline}
    return object_dict


@hydra.main(
    version_base="1.4",
    config_path="../configs",
    config_name="generate_audio_patch.yaml",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # if accelerator.is_main_process:
    extras(cfg)
    generate_with_patch(cfg, accelerator=accelerator)


if __name__ == "__main__":
    main()
