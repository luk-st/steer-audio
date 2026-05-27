from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from src.models.diffrhythm.modeling_diffrhythm import NNSightDiffRhythm, NNsightDiffRhythmDiffusionModel
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PatchableDiffRhythm:
    def __init__(
        self,
        max_frames: int = 2048,
        device: str | None = None,
    ):
        self.module = NNSightDiffRhythm(repo_id="ASLP-lab/DiffRhythm-base", max_frames=max_frames)
        self.patchable_model = NNsightDiffRhythmDiffusionModel(
            self.module, "ASLP-lab/DiffRhythm-base", max_frames=max_frames
        )
        if device is not None:
            self.patchable_model = self.patchable_model.to(device)
        self.patchable_model._model.pipeline.vae_model = self.patchable_model._model.pipeline.vae_model.to(device)
        self.duration = max_frames
        self.pipeline = self.patchable_model._model.pipeline

    def get_layers(self, layers_names: list[str]):
        return [(n, m) for (n, m) in self.patchable_model.named_modules() if n in layers_names]

    def prepare_latents(
        self,
        n_prompts: int,
        seed: int = 42,
    ):
        latents = self.pipeline.prepare_latents(n_prompts=n_prompts, duration=self.duration, seed=seed)
        return latents

    def _generate_clean_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        lrc_prompts: Union[list[str], str] = """""",
        num_inference_steps: int = 32,
        guidance_scale: float = 4.0,
    ):
        collected_activations = defaultdict(list)
        layers = self.get_layers(layers_to_patch)

        with self.patchable_model.generate(
            prompts_batch,
            latents=latents_batch,
            max_frames=self.duration,
            lrc_prompts=lrc_prompts,
            diffusion_steps=num_inference_steps,
            cfg=guidance_scale,
            trace=True,
        ):
            for _ in range(num_inference_steps-1):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]
                    collected_activations[layer_name].append(layer.inputs[0][0].cpu().save())
                    layer = layer.next()
                    layers[layer_idx] = (layer_name, layer)

                if guidance_scale >= 1e-5:
                    # null predictions
                    for layer_idx in range(len(layers)):
                        layer_name, layer = layers[layer_idx]
                        collected_activations["null_" + layer_name].append(layer.inputs[0][0].cpu().save())
                        layer = layer.next()
                        layers[layer_idx] = (layer_name, layer)

            outputs = self.patchable_model.output.save()

        return {
            "activations": collected_activations,
            "outputs": outputs,
        }

    def _generate_patched_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        device: torch.device,
        activations_batch: dict[str, list[torch.Tensor]],
        lrc_prompts: Union[list[str], str] = """""",
        num_inference_steps: int = 32,
        guidance_scale: float = 4.0,
        is_first_batch: bool = False,
    ):
        layers = self.get_layers(layers_to_patch)
        n_patches = 0

        with self.patchable_model.generate(
            prompts_batch,
            latents=latents_batch,
            max_frames=self.duration,
            lrc_prompts=lrc_prompts,
            diffusion_steps=num_inference_steps,
            cfg=guidance_scale,
            trace=True,
        ):
            for ts_idx in range(num_inference_steps-1):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]
                    # clean patching
                    layer.inputs = ((activations_batch[layer_name][ts_idx].to(device),),{})
                    layer = layer.next()
                    n_patches += 1
                    layers[layer_idx] = (layer_name, layer)

                if guidance_scale >= 1e-5:
                    # null patching
                    for layer_idx in range(len(layers)):
                        layer_name, layer = layers[layer_idx]
                        layer.inputs = ((activations_batch["null_"+layer_name][ts_idx].to(device),),{})
                        layer = layer.next()
                        n_patches += 1
                        layers[layer_idx] = (layer_name, layer)

            outputs = self.patchable_model.output.save()

        if is_first_batch:
            log.info(f"Patched activations n={n_patches} times")
        return {
            "outputs": outputs,
        }

    def _generate_ablated_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        lrc_prompts: Union[list[str], str] = """""",
        num_inference_steps: int = 32,
        guidance_scale: float = 4.0,
        is_first_batch: bool = False,
        ablate_null_pred: bool = False,
    ):
        layers = self.get_layers(layers_to_patch)
        n_patches = 0

        with self.patchable_model.generate(
            prompts_batch,
            latents=latents_batch,
            max_frames=self.duration,
            lrc_prompts=lrc_prompts,
            diffusion_steps=num_inference_steps,
            cfg=guidance_scale,
            trace=True,
        ):
            # outputs = nnsight.dict().save()
            # for ln in layers_to_patch:
            #     outputs[ln] = nnsight.list().save()
            for _ in range(num_inference_steps-1):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]
                    # outputs[layer_name].append(layer.output)
                    layer.output[0][:] = torch.zeros_like(layer.output[0])
                    n_patches += 1
                    layer = layer.next()
                    layers[layer_idx] = (layer_name, layer)

                if guidance_scale >= 1e-5:
                    # null patching
                    for layer_idx in range(len(layers)):
                        layer_name, layer = layers[layer_idx]
                        if ablate_null_pred is True:
                            layer.output[0][:] = torch.zeros_like(layer.output[0])
                            # outputs[layer_name].append(layer.output)
                            n_patches += 1
                        layer = layer.next()
                        layers[layer_idx] = (layer_name, layer)
            results = self.patchable_model.output.save()

        if is_first_batch:
            print(f"Ablated activations n={n_patches} times")
        return {
            "outputs": results,
            # "saved": outputs
        }

    def generate_by_patching(
        self,
        prompts_clean: list[str],
        prompts_corrupted: list[str],
        layers_to_patch: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        lrc_prompts: Union[list[str], str] = """""",
        num_inference_steps: int = 32,
        guidance_scale: float = 4.0,
    ):
        if latents.shape[0] != len(prompts_clean):
            raise ValueError(f"Latents shape {latents.shape} does not match number of prompts {len(prompts_clean)}")
        if latents.shape[0] != len(prompts_corrupted):
            raise ValueError(
                f"Latents shape {latents.shape} does not match number of prompts {len(prompts_corrupted)}"
            )

        batch_loop_base = range(0, len(prompts_clean), batch_size)
        batch_loop_cache = (
            tqdm(batch_loop_base, desc="Batched caching/patching") if accelerator.is_main_process else batch_loop_base
        )

        if accelerator.is_main_process:
            temp_layers = [n for n, _ in self.get_layers(layers_to_patch)]
            log.info(f"{len(temp_layers)} layers to patch: {temp_layers}")

        outputs_clean = []
        outputs_patched = []
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            prompts_corrupted_batch = prompts_corrupted[batch_idx_start:batch_idx_end]
            latents_batch = latents[batch_idx_start:batch_idx_end]

            clean_batch_result = self._generate_clean_batch(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                lrc_prompts=lrc_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            patched_batch_result = self._generate_patched_batch(
                prompts_batch=prompts_corrupted_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                device=accelerator.device,
                activations_batch=clean_batch_result["activations"],
                lrc_prompts=lrc_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                is_first_batch=batch_idx_start == 0,
            )
            outputs_clean.append(clean_batch_result["outputs"].cpu().numpy())
            outputs_patched.append(patched_batch_result["outputs"].cpu().numpy())

        outputs_clean = np.concatenate(outputs_clean, axis=0)
        outputs_patched = np.concatenate(outputs_patched, axis=0)
        log.info(f"Caching/patching done, n={len(outputs_clean)}")
        return {"clean": outputs_clean, "patched": outputs_patched}

    def generate_by_ablating(
        self,
        prompts_clean: list[str],
        layers_to_ablate: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        lrc_prompts: Union[list[str], str] = """""",
        num_inference_steps: int = 32,
        guidance_scale: float = 4.0,
        ablate_null_pred: bool = False,
    ):
        if latents.shape[0] != len(prompts_clean):
            raise ValueError(f"Latents shape {latents.shape} does not match number of prompts {len(prompts_clean)}")

        batch_loop_base = range(0, len(prompts_clean), batch_size)
        batch_loop_cache = (
            tqdm(batch_loop_base, desc="Batched ablating") if accelerator.is_main_process else batch_loop_base
        )

        if accelerator.is_main_process:
            temp_layers = [n for n, _ in self.get_layers(layers_to_ablate)]
            log.info(f"{len(temp_layers)} layers to ablate: {temp_layers}")

        outputs_ablated = []
        # outputs_collected = None
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            latents_batch = latents[batch_idx_start:batch_idx_end]

            ablated_batch_result = self._generate_ablated_batch(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_ablate,
                lrc_prompts=lrc_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                is_first_batch=batch_idx_start == 0,
                ablate_null_pred=ablate_null_pred,
            )
            outputs_ablated.append(ablated_batch_result["outputs"].cpu().numpy())
            # outputs_collected = ablated_batch_result["saved"]

        outputs_ablated = np.concatenate(outputs_ablated, axis=0)
        log.info(f"Ablating done, n={len(outputs_ablated)}")
        return {
            "ablated": outputs_ablated,
            # "saved": outputs_collected
        }