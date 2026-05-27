from collections import defaultdict
from pathlib import Path
from typing import Any
from src.models.utils import move_tensor_obj_to_device

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AudioLDMPipeline
from nnsight.modeling.diffusion import DiffusionModel
from tqdm import tqdm

from src.models.audioldm.utils import get_cross_attention_inputs_keys
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
import nnsight


class PatchableAudioLDM2:
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str | None = None,
        negative_prompt: str | None = None,
    ):
        self.model = DiffusionModel("cvssp/audioldm2-large", torch_dtype=dtype, dispatch=True)
        if device is not None:
            self.model = self.model.to(device)
        self.model.pipeline.set_progress_bar_config(disable=True)
        self.pipeline: AudioLDMPipeline = self.model.pipeline
        self.negative_prompt = negative_prompt

    def get_layers(self, layers_names: list[str]):
        return [(n, m) for (n, m) in self.model.named_modules() if n in layers_names]

    def get_layer(self, layer_name: str):
        return [m for (n, m) in self.model.named_modules() if n == layer_name][0]

    def _prepare_latent_height(self, audio_length_in_s: float | None = None):
        vocoder_upsample_factor = (
            np.prod(self.pipeline.vocoder.config.upsample_rates) / self.pipeline.vocoder.config.sampling_rate
        )

        if audio_length_in_s is None:
            audio_length_in_s = (
                self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor * vocoder_upsample_factor
            )

        height = int(audio_length_in_s / vocoder_upsample_factor)
        if height % self.pipeline.vae_scale_factor != 0:
            height = int(np.ceil(height / self.pipeline.vae_scale_factor)) * self.pipeline.vae_scale_factor

        return height

    def prepare_latents(
        self,
        n_prompts: int,
        seed: int = 42,
        audio_length_in_s: float | None = None,
    ):
        generator = torch.Generator(self.model.pipeline.device).manual_seed(seed)

        latents = self.pipeline.prepare_latents(
            batch_size=n_prompts,
            num_channels_latents=self.pipeline.unet.config.in_channels,
            height=self._prepare_latent_height(audio_length_in_s=audio_length_in_s),
            dtype=self.pipeline.dtype,
            device=torch.device(self.pipeline.device),
            generator=generator,
            latents=None,
        )
        return latents

    def _generate_clean_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        num_inference_steps: int = 200,
        audio_length_in_s: float | None = None,
        guidance_scale: float = 5.0,
        negative_prompt: list[str] | None = None,
    ):
        collected_activations = defaultdict(list)
        layers = self.get_layers(layers_to_patch)

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_length_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            latents=latents_batch,
            trace=True,
        ):
            for _ in range(num_inference_steps):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]
                    inputs_to_save = get_cross_attention_inputs_keys(layer_name)

                    saved_inputs = {}
                    for input_name in inputs_to_save:
                        saved_input = layer.inputs[1][input_name].save()
                        saved_inputs[input_name] = saved_input
                    collected_activations[layer_name].append(saved_inputs)

                    layer = layer.next()
                    layers[layer_idx] = (layer_name, layer)
            outputs = self.model.output.save()

        return {
            "activations": collected_activations,
            "outputs": outputs.audios,
        }

    def _generate_patched_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        device: torch.device,
        activations_batch: dict[str, list[dict[str, torch.Tensor]]],
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 5.0,
        negative_prompt: list[str] | None = None,
        is_first_batch: bool = False,
    ):
        layers = self.get_layers(layers_to_patch)
        n_patches = 0

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_length_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            latents=latents_batch,
            trace=True,
        ):
            for ts_idx in range(num_inference_steps):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]
                    layer_activations = activations_batch[layer_name][ts_idx]
                    for input_name, activations in layer_activations.items():
                        layer_input = activations
                        n_patches += 1
                        layer.inputs[1][input_name] = layer_input
                    layer = layer.next()
                    layers[layer_idx] = (layer_name, layer)
            outputs = self.model.output.save()

        if is_first_batch:
            log.info(f"Patched activations n={n_patches} times")
        return {
            "outputs": outputs.audios,
        }

    def _generate_ablate_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_ablate: list[str],
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 5.0,
        negative_prompts_batch: list[str] | None = None,
        is_first_batch: bool = False,
        seed: int = 42,
        ablate_null_pred: bool = False,
    ):
        split_cfg_start = len(prompts_batch) if (guidance_scale > 1 and ablate_null_pred is False) else 0

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_length_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts_batch,
            latents=latents_batch,
            seed=seed,
            trace=True,
        ):
            with self.model.unet.all():
                for layer_name in layers_to_ablate:
                    layer = self.get_layer(layer_name=layer_name)
                    layer.output[0][split_cfg_start:] = torch.zeros_like(layer.output[0][split_cfg_start:])
            audios = self.model.output.save()

        return {
            "outputs": audios.audios,
        }

    def _generate_inputs_outputs(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        num_inference_steps: int = 200,
        audio_length_in_s: float | None = None,
        guidance_scale: float = 5.0,
        negative_prompt: list[str] | None = None,
    ):
        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_length_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            latents=latents_batch,
            trace=True,
        ):
            inputs = nnsight.dict().save()  # type: ignore
            outputs = nnsight.dict().save()  # type: ignore
            for layer_name in layers_to_patch:
                inputs[layer_name] = nnsight.list().save()  # type: ignore
                outputs[layer_name] = nnsight.list().save()  # type: ignore

            with self.model.unet.all():  # type: ignore
                for layer_name in layers_to_patch:
                    layer = self.get_layer(layer_name=layer_name)
                    layer_inputs = layer.inputs.save()
                    inputs[layer_name].append(move_tensor_obj_to_device(layer_inputs, "cpu"))
                    layer_outputs = layer.output.save()
                    outputs[layer_name].append(move_tensor_obj_to_device(layer_outputs, "cpu"))
            audios = self.model.output.save()

        return {
            "inputs": inputs,
            "outputs": outputs,
            "audios": audios,
        }

    def generate_by_patching(
        self,
        prompts_clean: list[str],
        prompts_corrupted: list[str],
        layers_to_patch: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 5.0,
    ):
        if latents.shape[0] != len(prompts_clean):
            raise ValueError(f"Latents shape {latents.shape} does not match number of prompts {len(prompts_clean)}")
        if latents.shape[0] != len(prompts_corrupted):
            raise ValueError(
                f"Latents shape {latents.shape} does not match number of prompts {len(prompts_corrupted)}"
            )

        if self.negative_prompt is not None:
            negative_prompt = [self.negative_prompt] * len(prompts_clean)
        else:
            negative_prompt = None

        batch_loop_base = range(0, len(prompts_clean), batch_size)
        batch_loop_cache = (
            tqdm(batch_loop_base, desc="Batched caching/patching") if accelerator.is_main_process else batch_loop_base
        )

        if accelerator.is_main_process:
            temp_layers = [n for n, _ in self.get_layers(layers_to_patch)]
            log.info(f"{len(temp_layers)} layers to patch: {temp_layers}")
            for layer_name in temp_layers:
                log.info(f"Layer {layer_name} inputs to gather: {get_cross_attention_inputs_keys(layer_name)}")

        outputs_clean = []
        outputs_patched = []
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            prompts_corrupted_batch = prompts_corrupted[batch_idx_start:batch_idx_end]
            neg_prompts_batch = negative_prompt[batch_idx_start:batch_idx_end] if negative_prompt is not None else None
            latents_batch = latents[batch_idx_start:batch_idx_end]

            clean_batch_result = self._generate_clean_batch(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=audio_length_in_s,
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompts_batch,
            )
            patched_batch_result = self._generate_patched_batch(
                prompts_batch=prompts_corrupted_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                device=accelerator.device,
                num_inference_steps=num_inference_steps,
                activations_batch=clean_batch_result["activations"],
                audio_length_in_s=audio_length_in_s,
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompts_batch,
                is_first_batch=batch_idx_start == 0,
            )
            outputs_clean.append(clean_batch_result["outputs"])
            outputs_patched.append(patched_batch_result["outputs"])

        outputs_clean = np.concatenate(outputs_clean, axis=0)
        outputs_patched = np.concatenate(outputs_patched, axis=0)
        return {"clean": outputs_clean, "patched": outputs_patched}

    def get_inputs_outputs(
        self,
        prompts_clean: list[str],
        layers_to_patch: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 5.0,
    ):
        if latents.shape[0] != len(prompts_clean):
            raise ValueError(f"Latents shape {latents.shape} does not match number of prompts {len(prompts_clean)}")

        if self.negative_prompt is not None:
            negative_prompt = [self.negative_prompt] * len(prompts_clean)
        else:
            negative_prompt = None

        batch_loop_base = range(0, len(prompts_clean), batch_size)
        batch_loop_cache = (
            tqdm(batch_loop_base, desc="Batched collection") if accelerator.is_main_process else batch_loop_base
        )

        if accelerator.is_main_process:
            temp_layers = [n for n, _ in self.get_layers(layers_to_patch)]
            log.info(f"{len(temp_layers)} layers to patch: {temp_layers}")
            for layer_name in temp_layers:
                log.info(f"Layer {layer_name} inputs to gather: {get_cross_attention_inputs_keys(layer_name)}")

        inputs = defaultdict(list)
        outputs = defaultdict(list)
        audios = []
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            neg_prompts_batch = negative_prompt[batch_idx_start:batch_idx_end] if negative_prompt is not None else None
            latents_batch = latents[batch_idx_start:batch_idx_end]

            inputs_outputs_batch = self._generate_inputs_outputs(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=audio_length_in_s,
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompts_batch,
            )
            for layer_name in layers_to_patch:
                inputs[layer_name].append(inputs_outputs_batch["inputs"][layer_name])
                outputs[layer_name].append(inputs_outputs_batch["outputs"][layer_name])
            audios.append(inputs_outputs_batch["audios"].audios)

        audios = np.concatenate(audios, axis=0)

        return {
            "inputs": inputs,
            "outputs": outputs,
            "audios": audios,
        }

    def generate_by_ablating(
        self,
        prompts_clean: list[str],
        layers_to_ablate: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 5.0,
        seed: int = 42,
        ablate_null_pred: bool = False,
    ):
        if latents.shape[0] != len(prompts_clean):
            raise ValueError(f"Latents shape {latents.shape} does not match number of prompts {len(prompts_clean)}")

        if self.negative_prompt is not None:
            negative_prompt = [self.negative_prompt] * len(prompts_clean)
        else:
            negative_prompt = None

        batch_loop_base = range(0, len(prompts_clean), batch_size)
        batch_loop_cache = (
            tqdm(batch_loop_base, desc="Batched ablating") if accelerator.is_main_process else batch_loop_base
        )

        if accelerator.is_main_process:
            temp_layers = [n for n, _ in self.get_layers(layers_to_ablate)]
            log.info(f"{len(temp_layers)} layers to ablate: {temp_layers}")

        outputs_ablated = []
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            neg_prompts_batch = negative_prompt[batch_idx_start:batch_idx_end] if negative_prompt is not None else None
            latents_batch = latents[batch_idx_start:batch_idx_end]

            ablated_batch_result = self._generate_ablate_batch(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_ablate=layers_to_ablate,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=audio_length_in_s,
                guidance_scale=guidance_scale,
                negative_prompts_batch=neg_prompts_batch,
                is_first_batch=batch_idx_start == 0,
                ablate_null_pred=ablate_null_pred,
                seed=seed,
            )
            outputs_ablated.append(ablated_batch_result["outputs"])

        outputs_ablated = np.concatenate(outputs_ablated, axis=0)
        return {"ablated": outputs_ablated}
