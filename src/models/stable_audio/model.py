from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from nnsight.modeling.diffusion import DiffusionModel
from diffusers import StableAudioPipeline
from tqdm import tqdm

from src.models.stable_audio.utils import get_cross_attention_inputs_keys, should_patch_kv_inputs
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PatchableStableAudio:
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str | None = None,
        negative_prompt: str | None = None,
    ):
        self.model = DiffusionModel("stabilityai/stable-audio-open-1.0", torch_dtype=dtype, dispatch=True)
        if device is not None:
            self.model = self.model.to(device)
        self.model.pipeline.set_progress_bar_config(disable=True)
        self.pipeline: StableAudioPipeline = self.model.pipeline
        self.negative_prompt = negative_prompt

    def get_layers(self, layers_names: list[str]):
        return [(n, m) for (n, m) in self.model.named_modules() if n in layers_names]

    def get_layer(self, layer_name: str):
        return [m for (n, m) in self.model.named_modules() if n==layer_name][0]

    def postprocess_audio(self, audios: torch.Tensor) -> torch.Tensor:
        audios = audios.cpu()
        return audios.to(torch.float32).div(torch.max(torch.abs(audios)))

    def prepare_latents(self, n_prompts: int, seed: int = 42, **kwargs) -> torch.Tensor:
        shape = (
            n_prompts,
            self.pipeline.transformer.config.in_channels,
            int(self.pipeline.transformer.config.sample_size),
        )
        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=torch.device("cpu"), dtype=self.pipeline.dtype)
        return latents

    def _generate_clean_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        negative_prompts_batch: list[str] | None = None,
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        seed: int = 42,
    ):
        collected_activations = defaultdict(list)
        layers = self.get_layers(layers_to_patch)

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_end_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts_batch,
            latents=latents_batch,
            seed=seed,
            trace=True,
        ):
            for _ in range(num_inference_steps):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]

                    if should_patch_kv_inputs(layer_name):
                        saved_inputs = layer.inputs[0][0].cpu().save()
                    else:
                        inputs_to_save = get_cross_attention_inputs_keys(layer_name)
                        saved_inputs = {}
                        for input_name in inputs_to_save:
                            saved_input = layer.inputs[1][input_name].save()
                            saved_inputs[input_name] = saved_input

                    collected_activations[layer_name].append(saved_inputs)

                    layer = layer.next()
                    layers[layer_idx] = (layer_name, layer)
            outputs = self.model.output.save()

        audio = self.postprocess_audio(outputs.audios)

        return {
            "activations": collected_activations,
            "outputs": audio,
        }

    def _generate_patched_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        device: torch.device,
        activations_batch: dict[str, list[torch.Tensor]],
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompts_batch: list[str] | None = None,
        is_first_batch: bool = False,
        seed: int = 42,
    ):
        layers = self.get_layers(layers_to_patch)
        n_patches = 0

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_end_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts_batch,
            latents=latents_batch,
            seed=seed,
            trace=True,
        ):
            for ts_idx in range(num_inference_steps):
                for layer_idx in range(len(layers)):
                    layer_name, layer = layers[layer_idx]
                    layer_activations = activations_batch[layer_name][ts_idx]

                    if should_patch_kv_inputs(layer_name):
                        layer.inputs = ((layer_activations.to(device),), {})
                        n_patches += 1
                    else:
                        for input_name in layer_activations.keys():
                            n_patches += 1
                            layer.inputs[1][input_name] = layer_activations[input_name]
                    layer = layer.next()
                    layers[layer_idx] = (layer_name, layer)
            outputs = self.model.output.save()

        audio = self.postprocess_audio(outputs.audios)

        if is_first_batch:
            log.info(f"Patched activations n={n_patches} times")
        return {"outputs": audio}

    def generate_by_patching(
        self,
        prompts_clean: list[str],
        prompts_corrupted: list[str],
        layers_to_patch: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        seed: int = 42,
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
                negative_prompts_batch=neg_prompts_batch,
                seed=seed,
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
                negative_prompts_batch=neg_prompts_batch,
                seed=seed,
                is_first_batch=batch_idx_start == 0,
            )
            outputs_clean.append(clean_batch_result["outputs"])
            outputs_patched.append(patched_batch_result["outputs"])

        outputs_clean = np.concatenate(outputs_clean, axis=0)
        outputs_patched = np.concatenate(outputs_patched, axis=0)
        return {"clean": outputs_clean, "patched": outputs_patched}

    def _generate_ablate_transformers_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_ablate: list[str],
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompts_batch: list[str] | None = None,
        is_first_batch: bool = False,
        seed: int = 42,
        ablate_null_pred: bool = False,
    ):
        n_ablates = 0
        split_cfg_start = len(prompts_batch) if (guidance_scale > 1 and ablate_null_pred is False) else 0

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_end_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts_batch,
            latents=latents_batch,
            seed=seed,
            trace=True,
        ):
            with self.model.transformer.all():
                for ln in layers_to_ablate:
                    layer = self.get_layer(ln)
                    patch_value = torch.zeros_like(layer.output[split_cfg_start:]) if ln.endswith("attn1") else layer.inputs[1]["hidden_states"][split_cfg_start:]
                    layer.output[split_cfg_start:] = patch_value
                    n_ablates += 1

            outputs = self.model.output.save()
        audio = self.postprocess_audio(outputs.audios)

        if is_first_batch:
            log.info(f"Ablated activations n={n_ablates} times")
            log.info(f"Ablating from index: {split_cfg_start}")
        return {"outputs": audio}

    def _generate_ablate_attn2_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_ablate: list[str],
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompts_batch: list[str] | None = None,
        is_first_batch: bool = False,
        seed: int = 42,
        ablate_null_pred: bool = False,
    ):
        n_ablates = 0
        split_cfg_start = len(prompts_batch) if (guidance_scale > 1 and ablate_null_pred is False) else 0

        with self.model.generate(
            prompts_batch,
            num_inference_steps=num_inference_steps,
            num_waveforms_per_prompt=1,
            audio_end_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts_batch,
            latents=latents_batch,
            seed=seed,
            trace=True,
        ):
            with self.model.transformer.all():
                for ln in layers_to_ablate:
                    layer = self.get_layer(ln)
                    expected_output_shape = layer.output[split_cfg_start:]
                    layer.output[split_cfg_start:] = torch.zeros_like(expected_output_shape)
                    n_ablates += 1

            outputs = self.model.output.save()
        audio = self.postprocess_audio(outputs.audios)
 
        if is_first_batch:
            log.info(f"Ablated activations n={n_ablates} times")
        return {"outputs": audio}


    def generate_by_ablating(
            self,
            prompts_clean: list[str],
            layers_to_ablate: list[str],
            latents: torch.Tensor,
            batch_size: int,
            accelerator: Accelerator,
            audio_length_in_s: float | None = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 7.0,
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

                ablated_batch_result = self._generate_ablate_transformers_batch(
                    prompts_batch=prompts_clean_batch,
                    latents_batch=latents_batch,
                    layers_to_ablate=layers_to_ablate,
                    num_inference_steps=num_inference_steps,
                    audio_length_in_s=audio_length_in_s,
                    guidance_scale=guidance_scale,
                    negative_prompts_batch=neg_prompts_batch,
                    seed=seed,
                    is_first_batch=batch_idx_start == 0,
                    ablate_null_pred=ablate_null_pred,
                )
                outputs_ablated.append(ablated_batch_result["outputs"])

            outputs_ablated = np.concatenate(outputs_ablated, axis=0)
            return {"ablated": outputs_ablated}