from typing import Union

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from collections import defaultdict
from src.models.utils import move_tensor_obj_to_device

from src.models.ace_step.modeling_ace import NNSightSimpleACEStep, NNSightSimpleACEStepModel
from src.utils import RankedLogger
import nnsight
from src.models.ace_step.constants import CROSS_ATTENTION_PATCH_INPUT_KEYS, CROSS_ATTENTION_LAYERS, CROSS_ATTENTION_KV_OUTPUT_KEYS, CROSS_ATTENTION_KV_LAYERS

log = RankedLogger(__name__, rank_zero_only=True)

class PatchableACE:
    def __init__(
        self,
        device: str | None = None,
        pad_to_max_len: int | None = None,
    ):
        self.module = NNSightSimpleACEStep(device = device, dtype = "bfloat16", pad_to_max_len=pad_to_max_len)
        self.patchable_model = NNSightSimpleACEStepModel(
            self.module, ""
        )
        self.pipeline = self.patchable_model._model.pipeline

    def get_layers(self, layers_names: list[str]):
        return [(n, m) for (n, m) in self.patchable_model.named_modules() if n in layers_names]

    def get_layer_by_name(self, layer_name: str):
        return [m for (n,m) in self.patchable_model.ace_step_transformer.named_modules() if n == layer_name][0]

    def prepare_latents(
        self,
        n_prompts: int,
        audio_duration: float,
        seed: int = 42,
    ):
        latents = self.pipeline.prepare_latents(batch_size=n_prompts, audio_duration=audio_duration, seed=seed)
        return latents

    def _generate_clean_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        collectable_keys: dict[str, dict[str, list[str] | None]],
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
    ):
        with self.patchable_model.generate(
            prompts_batch,
            audio_duration=audio_duration,
            lyrics=lyrics,
            infer_step=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seed=seed,
            latents=latents_batch,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale_text=guidance_scale_text,
            guidance_interval=guidance_interval,
            trace=True,
        ):
            collected_activations = nnsight.dict().save()
            for ln in layers_to_patch:
                collected_activations[ln] = nnsight.list().save()

            with self.patchable_model.all():
                for ln in layers_to_patch:
                    layer = self.get_layer_by_name(ln)
                    if "inputs" in collectable_keys[ln].keys():
                        if collectable_keys[ln]["inputs"] is None:
                            traced_items = layer.inputs.save()
                        else:
                            traced_items = {}
                            for k in collectable_keys[ln]["inputs"]:
                                traced_items[k]=layer.inputs[1][k].save()
                    elif "outputs" in collectable_keys[ln].keys():
                        traced_items = layer.output.save()
                    else:
                        traced_items = None
                    collected_activations[ln].append(traced_items)
            outputs = self.patchable_model.output.save()

        n_to_patch = len(collected_activations[layers_to_patch[0]]) if len(layers_to_patch) > 0 else 0

        return {
            "activations": collected_activations,
            "outputs": outputs,
            "n_to_patch": n_to_patch,
        }

    def _generate_patched_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_patch: list[str],
        n_patch_iterations: int,
        collected_activations: dict[str, list[dict[str, torch.Tensor]]],
        collectable_keys: dict[str, dict[str, list[str] | None]],
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
    ):
        with self.patchable_model.generate(
            prompts_batch,
            audio_duration = audio_duration,
            lyrics = lyrics,
            infer_step = num_inference_steps,
            guidance_scale = guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seed=seed,
            latents=latents_batch,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale_text=guidance_scale_text,
            guidance_interval=guidance_interval,
            trace=True
        ):
            layers = {
                ln: self.get_layer_by_name(ln)
                for ln in layers_to_patch
            }
            for _ in range(n_patch_iterations):
                for ln in layers_to_patch:
                    layer = layers[ln]
                    activations = collected_activations[ln].pop(0)
                    if "inputs" in collectable_keys[ln].keys():
                        if collectable_keys[ln]["inputs"] is None:
                            layer.inputs = activations
                        else:
                            for k in collectable_keys[ln]["inputs"]:
                                layer.inputs[1][k] = activations[k]
                    elif "outputs" in collectable_keys[ln].keys():
                        layer.output[:] = activations
                    layer = layer.next()
                    layers[ln] = layer
            outputs = self.patchable_model.output.save()

        return {
            "outputs": outputs,
        }

    def _generate_ablated_batch(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_ablate: list[str],
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
    ):
        with self.patchable_model.generate(
            prompts_batch,
            audio_duration = audio_duration,
            lyrics = lyrics,
            infer_step = num_inference_steps,
            guidance_scale = guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seed=seed,
            latents=latents_batch,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale_text=guidance_scale_text,
            guidance_interval=guidance_interval,
            trace=True
        ):
            # l_inputs = nnsight.list().save()
            # l_outputs = nnsight.list().save()
            with self.patchable_model.all():
                for ln in layers_to_ablate:
                    layer = self.get_layer_by_name(ln)
                    layer.output[0][:] *= 0
                    # l_inputs.append(layer.inputs.save())
                    # l_outputs.append(layer.output.save())
            outputs = self.patchable_model.output.save()

        return {
            # "l_inputs": l_inputs,
            # "l_outputs": l_outputs,
            "outputs": outputs,
        }

    def _generate_inputs_outputs(
        self,
        prompts_batch: list[str],
        latents_batch: torch.Tensor,
        layers_to_collect: list[str],
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
    ):
        with self.patchable_model.generate(
            prompts_batch,
            audio_duration = audio_duration,
            lyrics = lyrics,
            infer_step = num_inference_steps,
            guidance_scale = guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seed=seed,
            latents=latents_batch,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale_text=guidance_scale_text,
            guidance_interval=guidance_interval,
            trace=True
        ):
            inputs = nnsight.dict().save()  # type: ignore
            outputs = nnsight.dict().save()  # type: ignore
            for layer_name in layers_to_collect:
                inputs[layer_name] = nnsight.list().save()  # type: ignore
                outputs[layer_name] = nnsight.list().save()  # type: ignore

            with self.patchable_model.all():  # type: ignore
                for layer_name in layers_to_collect:
                    layer = self.get_layer_by_name(layer_name=layer_name)
                    layer_inputs = layer.inputs.save()
                    inputs[layer_name].append(move_tensor_obj_to_device(layer_inputs, "cpu"))
                    layer_outputs = layer.output.save()
                    outputs[layer_name].append(move_tensor_obj_to_device(layer_outputs, "cpu"))
            audios = self.patchable_model.output.save()

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
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
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

        collectable_keys = {}
        for ln in layers_to_patch:
            if ln in CROSS_ATTENTION_LAYERS:
                collectable_keys[ln] = {"inputs": CROSS_ATTENTION_PATCH_INPUT_KEYS}
            elif ln in CROSS_ATTENTION_KV_LAYERS:
                collectable_keys[ln] = {"outputs": CROSS_ATTENTION_KV_OUTPUT_KEYS}
            else:
                raise ValueError(f"Layer {ln} is not a cross-attention/kv layer")

        outputs_clean = []
        outputs_patched = []
        batch_seed = seed
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            prompts_corrupted_batch = prompts_corrupted[batch_idx_start:batch_idx_end]
            latents_batch = latents[batch_idx_start:batch_idx_end]

            clean_batch_result = self._generate_clean_batch(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                collectable_keys=collectable_keys,
                audio_duration=audio_duration,
                lyrics=lyrics,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_lyric=guidance_scale_lyric,
                guidance_scale_text=guidance_scale_text,
                guidance_interval=guidance_interval,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                seed=batch_seed
            )

            patched_batch_result = self._generate_patched_batch(
                prompts_batch=prompts_corrupted_batch,
                latents_batch=latents_batch,
                layers_to_patch=layers_to_patch,
                n_patch_iterations=clean_batch_result["n_to_patch"],
                collected_activations=clean_batch_result["activations"],
                collectable_keys=collectable_keys,
                audio_duration=audio_duration,
                lyrics=lyrics,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_lyric=guidance_scale_lyric,
                guidance_scale_text=guidance_scale_text,
                guidance_interval=guidance_interval,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                seed=batch_seed
            )
            outputs_clean.append(clean_batch_result["outputs"].cpu().numpy())
            outputs_patched.append(patched_batch_result["outputs"].cpu().numpy())
            batch_seed += 1

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
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
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
        # l_inputs = []
        # l_outputs = []
        batch_seed = seed
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            latents_batch = latents[batch_idx_start:batch_idx_end]

            ablated_batch_result = self._generate_ablated_batch(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_ablate=layers_to_ablate,
                audio_duration=audio_duration,
                lyrics=lyrics,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_lyric=guidance_scale_lyric,
                guidance_scale_text=guidance_scale_text,
                guidance_interval=guidance_interval,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                seed=batch_seed
            )

            outputs_ablated.append(ablated_batch_result["outputs"].cpu().numpy())
            # l_inputs.append(ablated_batch_result["l_inputs"])
            # l_outputs.append(ablated_batch_result["l_outputs"])
            batch_seed += 1

        outputs_ablated = np.concatenate(outputs_ablated, axis=0)
        log.info(f"Ablation done, n={len(outputs_ablated)}")
        # return {"ablated": outputs_ablated, "l_inputs": l_inputs, "l_outputs": l_outputs}
        return {"ablated": outputs_ablated}

    def get_inputs_outputs(
        self,
        prompts_clean: list[str],
        layers_to_collect: list[str],
        latents: torch.Tensor,
        batch_size: int,
        accelerator: Accelerator,
        audio_duration: float = 60.0,
        lyrics: Union[list[str], str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_scale_lyric: float = 0.0,
        guidance_scale_text: float = 6.0,
        guidance_interval: float = 1.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 7.0,
        seed: int = 42,
    ):
        if latents.shape[0] != len(prompts_clean):
            raise ValueError(f"Latents shape {latents.shape} does not match number of prompts {len(prompts_clean)}")

        batch_loop_base = range(0, len(prompts_clean), batch_size)
        batch_loop_cache = (
            tqdm(batch_loop_base, desc="Batched collecting") if accelerator.is_main_process else batch_loop_base
        )

        if accelerator.is_main_process:
            temp_layers = [n for n, _ in self.get_layers(layers_to_collect)]
            log.info(f"{len(temp_layers)} layers to gather: {temp_layers}")

        inputs = defaultdict(list)
        outputs = defaultdict(list)
        audios = []
        batch_seed = seed
        for batch_idx_start in batch_loop_cache:
            batch_idx_end = batch_idx_start + batch_size
            prompts_clean_batch = prompts_clean[batch_idx_start:batch_idx_end]
            latents_batch = latents[batch_idx_start:batch_idx_end]

            inputs_outputs_batch = self._generate_inputs_outputs(
                prompts_batch=prompts_clean_batch,
                latents_batch=latents_batch,
                layers_to_collect=layers_to_collect,
                num_inference_steps=num_inference_steps,
                audio_duration=audio_duration,
                lyrics=lyrics,
                guidance_scale=guidance_scale,
                guidance_scale_lyric=guidance_scale_lyric,
                guidance_scale_text=guidance_scale_text,
                guidance_interval=guidance_interval,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                seed=batch_seed
            )
            for layer_name in layers_to_collect:
                inputs[layer_name].append(inputs_outputs_batch["inputs"][layer_name])
                outputs[layer_name].append(inputs_outputs_batch["outputs"][layer_name])
            audios.append(inputs_outputs_batch["audios"])
            batch_seed += 1

        audios = np.concatenate(audios, axis=0)

        return {
            "inputs": inputs,
            "outputs": outputs,
            "audios": audios,
        }