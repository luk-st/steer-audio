from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchaudio
from diffusers import DiffusionPipeline
from einops import rearrange
from IPython.display import clear_output
from muq import MuQMuLan
from nnsight import util
from nnsight.intervention.contexts import InterventionTracer
from nnsight.modeling.mixins import RemoteableMixin
from transformers import BatchEncoding
from typing_extensions import Self

from src.models.diffrhythm.DiffRhythm.infer.infer_utils import (
    CNENTokenizer,
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)
from src.models.diffrhythm.DiffRhythm.model.cfm import CFM
from src.models.diffrhythm.DiffRhythm.model.dit import DiT
from src.models.diffrhythm.utils import SuppressCStderr


class DiffRhythm:
    def __init__(self, repo_id: str = "ASLP-lab/DiffRhythm-base", max_frames: int = 2048, device: str = "cpu"):
        warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
        with SuppressCStderr():
            cfm, _, tokenizer, muq, vae = prepare_model(
                max_frames=max_frames, device=device, repo_id=repo_id
            )
        self.cfm_model: CFM = cfm
        self.tokenizer: CNENTokenizer = tokenizer
        self.muq_model: MuQMuLan = muq
        self.vae_model: torch.nn.Module = vae
        warnings.filterwarnings("default", category=UserWarning, module="onnxruntime")

    def prepare_latents(self, n_prompts: int, duration: int, seed: int = 42):
        return self.cfm_model.prepare_latents(n_prompts, duration, seed)

    def preprocess_lrc(
        self, lrc_prompts: list[str] | str, batch_size: int, max_frames: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(lrc_prompts, str):
            lrc, start_time = get_lrc_token(max_frames, lrc_prompts, self.tokenizer, device)
            lrc = lrc.repeat(batch_size, 1)
            start_time = start_time.repeat(batch_size)
        else:
            lrc_list = []
            start_time_list = []

            for lrc_prompt in lrc_prompts:
                lrc, start_time = get_lrc_token(max_frames, lrc_prompt, self.tokenizer, device)
                lrc_list.append(lrc)
                start_time_list.append(start_time)
            lrc = torch.cat(lrc_list, dim=0)
            start_time = torch.cat(start_time_list, dim=0)

        return lrc, start_time

    def decode_audio(self, latent: torch.Tensor, chunked: bool = False) -> torch.Tensor:
        output = decode_audio(latent, self.vae_model, chunked=chunked)
        output = (
            output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        )

        return output

    def generate_audio(
        self,
        prompts: list[str] | str,
        latents: torch.Tensor | None = None,
        max_frames: int = 2048,
        lrc_prompts: list[str] | str = "",
        cfg=4.0,
        diffusion_steps=32,
        seed: int = 42,
        device: str = "cuda",
        chunked=True,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        if latents is None:
            latents = self.prepare_latents(n_prompts=batch_size, duration=max_frames, seed=seed)
        else:
            assert latents.shape[0] == batch_size
        latents = latents.to(device)

        lrc, start_time = self.preprocess_lrc(
            lrc_prompts=lrc_prompts, batch_size=batch_size, max_frames=max_frames, device=device
        )
        style_prompts = get_style_prompt(self.muq_model, prompt=prompts)
        negative_style_prompts = get_negative_style_prompt(device).repeat(batch_size, 1)
        latent_prompt = get_reference_latent(device, max_frames).repeat(batch_size, 1, 1)

        with torch.inference_mode():
            generated, _ = self.cfm_model.sample(
                cond=latent_prompt,
                text=lrc,
                latents=latents,
                duration=max_frames,
                style_prompt=style_prompts,
                negative_style_prompt=negative_style_prompts,
                steps=diffusion_steps,
                cfg_strength=cfg,
                start_time=start_time,
            )

            generated = generated.to(torch.float32)
            latent = generated.transpose(1, 2)  # [b d t]
            output = self.decode_audio(latent=latent, chunked=chunked)

            return output


class NNSightDiffRhythm(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.pipeline = DiffRhythm(*args, **kwargs)
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module) and not isinstance(value, torch.jit.ScriptModule):
                setattr(self, key, value)

        self.tokenizer = self.pipeline.tokenizer


class NNsightDiffRhythmDiffusionModel(RemoteableMixin):
    __methods__ = {"generate": "_generate"}

    def __init__(self, *args, **kwargs) -> None:
        self._model: NNSightDiffRhythm = None
        super().__init__(*args, **kwargs)

    def _load_meta(self, repo_id: str, **kwargs):
        model = NNSightDiffRhythm(
            repo_id,
            **kwargs,
        )
        return model

    def _load(self, repo_id: str, **kwargs) -> NNSightDiffRhythm:

        model = NNSightDiffRhythm(repo_id, **kwargs)

        return model

    def _prepare_input(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:

        if isinstance(inputs, str):
            inputs = [inputs]

        return ((inputs,), {}), len(inputs)

    def _batch(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:

        if batched_inputs is None:

            return ((prepared_inputs,), {})

        return (batched_inputs + prepared_inputs,)

    def _execute(self, prepared_inputs: Any, *args, **kwargs):
        return self._model.cfm_model.transformer(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _generate(self, prepared_inputs: Any, *args, seed: int = None, **kwargs):
        if self._scanning():
            kwargs["num_inference_steps"] = 1
        generator = torch.Generator()

        if seed is not None:
            if isinstance(prepared_inputs, list):
                generator = [torch.Generator().manual_seed(seed) for _ in range(len(prepared_inputs))]
            else:
                generator = generator.manual_seed(seed)

        output = self._model.pipeline.generate_audio(prepared_inputs, *args, **kwargs)
        output = self._model(output)
        return output
