# based on https://github.com/ace-step/ACE-Step/blob/main/acestep/pipeline_ace_step.py

import os

import torch
import torchaudio
from diffusers.utils.torch_utils import randn_tensor

from .ACE.acestep.cpu_offload import cpu_offload
from .ACE.acestep.pipeline_ace_step import ACEStepPipeline

SAMPLE_RATE = 48000


class SimpleACEStepPipeline(ACEStepPipeline):
    def __init__(
        self,
        repo_id="",  # mock
        device="cpu",
        dtype="bfloat16",
        persistent_storage_path="res/ace_step",
        torch_compile=False,
        cpu_offload=False,
        quantized=False,
        overlapped_decode=False,
        pad_to_max_len=None,
    ):
        super().__init__(
            dtype=dtype,
            persistent_storage_path=persistent_storage_path,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            quantized=quantized,
            overlapped_decode=overlapped_decode,
            pad_to_max_len=pad_to_max_len,
        )
        self.device = device # type: ignore
        self.sample_rate = SAMPLE_RATE

    def load(self, lora_name_or_path: str = "none", lora_weight: float = 1.0):
        self.load_checkpoint(self.checkpoint_dir)
        self.load_lora(lora_name_or_path, lora_weight)

    def _get_audio_duration(self, audio_path: str) -> float:
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[1] > 240 * sr:
            return 240.0
        else:
            return audio.shape[1] / sr

    def task_edit_flowedit(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        retake_random_generators: list | None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        src_audio_path: str | None = None,
        edit_target_prompt: str | None = None,
        edit_target_lyrics: str | None = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        batch_size: int = 1,
        cfg_type: str = "apg",
        layers_to_hook: list[str] | None = None,
    ):
        # same as task_edit, but with the ability to hook certain layers
        src_latents = None
        assert src_audio_path is not None, "src_audio_path is required for edit task"
        assert os.path.exists(src_audio_path), (
            f"src_audio_path {src_audio_path} does not exist"
        )
        src_latents = self.infer_latents(src_audio_path)

        texts = [edit_target_prompt]
        target_encoder_text_hidden_states, target_text_attention_mask = (
            self.get_text_embeddings(texts)
        )
        target_encoder_text_hidden_states = target_encoder_text_hidden_states.repeat(
            batch_size, 1, 1
        )
        target_text_attention_mask = target_text_attention_mask.repeat(batch_size, 1)

        target_lyric_token_idx = (
            torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        )
        target_lyric_mask = (
            torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        )
        if len(edit_target_lyrics) > 0:
            target_lyric_token_idx = self.tokenize_lyrics(
                edit_target_lyrics, debug=True
            )
            target_lyric_mask = [1] * len(target_lyric_token_idx)
            target_lyric_token_idx = (
                torch.tensor(target_lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            target_lyric_mask = (
                torch.tensor(target_lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        target_speaker_embeds = speaker_embeds.clone()

        target_latents = self.flowedit_diffusion_process(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            target_encoder_text_hidden_states=target_encoder_text_hidden_states,
            target_text_attention_mask=target_text_attention_mask,
            target_speaker_embeds=target_speaker_embeds,
            target_lyric_token_ids=target_lyric_token_idx,
            target_lyric_mask=target_lyric_mask,
            src_latents=src_latents,
            random_generators=retake_random_generators,  # more diversity
            infer_steps=infer_step,
            guidance_scale=guidance_scale,
            n_min=edit_n_min,
            n_max=edit_n_max,
            n_avg=edit_n_avg,
            scheduler_type=scheduler_type,
            layers_to_hook=layers_to_hook,
        )

        return target_latents

    def task_edit_ode_inversion(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        retake_random_generators: list | None,
        infer_step: int = 60,
        tstart: int = 60,
        guidance_scale_inversion: float = 1.0,
        guidance_scale_editing: float = 10.0,
        scheduler_type: str = "euler",
        src_audio_path: str | None = None,
        edit_target_prompt: str | None = None,
        edit_target_lyrics: str | None = None,
        batch_size: int = 1,
        cfg_type: str = "apg",
        layers_to_hook: list[str] | None = None,
    ):
        src_latents = None
        assert src_audio_path is not None, "src_audio_path is required for edit task"
        assert os.path.exists(src_audio_path), (
            f"src_audio_path {src_audio_path} does not exist"
        )
        src_latents = self.infer_latents(src_audio_path)

        texts = [edit_target_prompt]
        target_encoder_text_hidden_states, target_text_attention_mask = (
            self.get_text_embeddings(texts)
        )
        target_encoder_text_hidden_states = target_encoder_text_hidden_states.repeat(
            batch_size, 1, 1
        )
        target_text_attention_mask = target_text_attention_mask.repeat(batch_size, 1)

        target_lyric_token_idx = (
            torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        )
        target_lyric_mask = (
            torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        )
        if len(edit_target_lyrics) > 0:
            target_lyric_token_idx = self.tokenize_lyrics(
                edit_target_lyrics, debug=False
            )
            target_lyric_mask = [1] * len(target_lyric_token_idx)
            target_lyric_token_idx = (
                torch.tensor(target_lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            target_lyric_mask = (
                torch.tensor(target_lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        target_speaker_embeds = speaker_embeds.clone()

        target_latents, all_inverted_latents = self.ode_inversion_invert(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            src_latents=src_latents,
            infer_steps=infer_step,
            tstart=tstart,
            guidance_scale_inversion=guidance_scale_inversion,
            cfg_type=cfg_type,
        )
        target_samples, all_sampled_latents = self.ode_inversion_sampling(
            source_encoder_text_hidden_states=encoder_text_hidden_states,
            source_text_attention_mask=text_attention_mask,
            source_speaker_embds=speaker_embeds,
            source_lyric_token_ids=lyric_token_idx,
            source_lyric_mask=lyric_mask,
            target_encoder_text_hidden_states=target_encoder_text_hidden_states,
            target_text_attention_mask=target_text_attention_mask,
            target_speaker_embds=target_speaker_embeds,
            target_lyric_token_ids=target_lyric_token_idx,
            target_lyric_mask=target_lyric_mask,
            src_latents=target_latents,
            infer_steps=infer_step,
            tstart=tstart,
            guidance_scale_sampling=guidance_scale_editing,
            cfg_type=cfg_type,
            layers_to_hook=layers_to_hook,
        )
        return {
            "target_latents": target_latents,
            "target_samples": target_samples,
            "all_inverted_latents": all_inverted_latents,
            "all_sampled_latents": all_sampled_latents,
        }

    def task_edit(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        retake_random_generators: list | None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        src_audio_path: str | None = None,
        edit_target_prompt: str | None = None,
        edit_target_lyrics: str | None = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        batch_size: int = 1,
    ):
        src_latents = None
        assert src_audio_path is not None, "src_audio_path is required for edit task"
        assert os.path.exists(src_audio_path), (
            f"src_audio_path {src_audio_path} does not exist"
        )
        src_latents = self.infer_latents(src_audio_path)

        texts = [edit_target_prompt]
        target_encoder_text_hidden_states, target_text_attention_mask = (
            self.get_text_embeddings(texts)
        )
        target_encoder_text_hidden_states = target_encoder_text_hidden_states.repeat(
            batch_size, 1, 1
        )
        target_text_attention_mask = target_text_attention_mask.repeat(batch_size, 1)

        target_lyric_token_idx = (
            torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        )
        target_lyric_mask = (
            torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        )
        if len(edit_target_lyrics) > 0:
            target_lyric_token_idx = self.tokenize_lyrics(
                edit_target_lyrics, debug=True
            )
            target_lyric_mask = [1] * len(target_lyric_token_idx)
            target_lyric_token_idx = (
                torch.tensor(target_lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            target_lyric_mask = (
                torch.tensor(target_lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        target_speaker_embeds = speaker_embeds.clone()

        target_latents = self.flowedit_diffusion_process(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            target_encoder_text_hidden_states=target_encoder_text_hidden_states,
            target_text_attention_mask=target_text_attention_mask,
            target_speaker_embeds=target_speaker_embeds,
            target_lyric_token_ids=target_lyric_token_idx,
            target_lyric_mask=target_lyric_mask,
            src_latents=src_latents,
            random_generators=retake_random_generators,  # more diversity
            infer_steps=infer_step,
            guidance_scale=guidance_scale,
            n_min=edit_n_min,
            n_max=edit_n_max,
            n_avg=edit_n_avg,
            scheduler_type=scheduler_type,
        )

        return target_latents

    def task_repaint(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        random_generators: list,
        encoder_text_hidden_states_null: torch.Tensor | None,
        retake_random_generators: list | None,
        audio_duration: float = 60.0,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str | None = None,
        retake_variance: float = 0.5,
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str | None = None,
    ):
        add_retake_noise = True
        src_latents = None

        assert src_audio_path is not None, "src_audio_path is required for repaint task"
        assert os.path.exists(src_audio_path), (
            f"src_audio_path {src_audio_path} does not exist"
        )
        src_latents = self.infer_latents(src_audio_path)

        assert ref_audio_input is None and audio2audio_enable is False, (
            "ref_audio_input and audio2audio_enable are not supported for repaint task"
        )
        ref_latents = None

        target_latents = self.text2music_diffusion_process(
            duration=audio_duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            infer_steps=infer_step,
            random_generators=random_generators,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            oss_steps=oss_steps,
            encoder_text_hidden_states_null=encoder_text_hidden_states_null,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            retake_random_generators=retake_random_generators,
            retake_variance=retake_variance,
            add_retake_noise=add_retake_noise,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_latents=src_latents,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_latents=ref_latents,
        )
        return target_latents

    def task_extend(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        random_generators: list,
        encoder_text_hidden_states_null: torch.Tensor | None,
        retake_random_generators: list | None,
        audio_duration: float = 60.0,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str | None = None,
        retake_variance: float = 0.5,
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str | None = None,
    ):
        add_retake_noise = True
        src_latents = None

        assert src_audio_path is not None, "src_audio_path is required for extend task"
        assert os.path.exists(src_audio_path), (
            f"src_audio_path {src_audio_path} does not exist"
        )
        src_latents = self.infer_latents(src_audio_path)

        assert ref_audio_input is None and audio2audio_enable is False, (
            "ref_audio_input and audio2audio_enable are not supported for extent task"
        )
        ref_latents = None

        target_latents = self.text2music_diffusion_process(
            duration=audio_duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            infer_steps=infer_step,
            random_generators=random_generators,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            oss_steps=oss_steps,
            encoder_text_hidden_states_null=encoder_text_hidden_states_null,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            retake_random_generators=retake_random_generators,
            retake_variance=retake_variance,
            add_retake_noise=add_retake_noise,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_latents=src_latents,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_latents=ref_latents,
        )

        return target_latents

    def task_retake(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        random_generators: list,
        encoder_text_hidden_states_null: torch.Tensor | None,
        retake_random_generators: list | None,
        audio_duration: float = 60.0,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str | None = None,
        retake_variance: float = 0.5,
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str | None = None,
    ):
        add_retake_noise = True
        # retake equal to repaint
        repaint_start = 0
        repaint_end = audio_duration

        assert src_audio_path is not None, (
            "src_audio_path is not required for retake task"
        )
        src_latents = None

        assert ref_audio_input is None and audio2audio_enable is False, (
            "ref_audio_input and audio2audio_enable are not supported for retake task"
        )
        ref_latents = None

        target_latents = self.text2music_diffusion_process(
            duration=audio_duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            infer_steps=infer_step,
            random_generators=random_generators,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            oss_steps=oss_steps,
            encoder_text_hidden_states_null=encoder_text_hidden_states_null,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            retake_random_generators=retake_random_generators,
            retake_variance=retake_variance,
            add_retake_noise=add_retake_noise,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_latents=src_latents,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_latents=ref_latents,
        )
        return target_latents

    def task_audio2audio(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        random_generators: list,
        encoder_text_hidden_states_null: torch.Tensor | None,
        retake_random_generators: list | None,
        audio_duration: float = 60.0,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str = None,
        retake_variance: float = 0.5,
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str | None = None,
    ):
        assert audio2audio_enable and ref_audio_input is not None, (
            "audio2audio_enable and ref_audio_input are required for audio2audio task"
        )

        add_retake_noise = False

        assert src_audio_path is None, (
            "src_audio_path is not supported for audio2audio task"
        )
        src_latents = None
        assert os.path.exists(ref_audio_input), (
            f"ref_audio_input {ref_audio_input} does not exist"
        )
        ref_latents = self.infer_latents(ref_audio_input)

        target_latents = self.text2music_diffusion_process(
            duration=audio_duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            infer_steps=infer_step,
            random_generators=random_generators,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            oss_steps=oss_steps,
            encoder_text_hidden_states_null=encoder_text_hidden_states_null,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            retake_random_generators=retake_random_generators,
            retake_variance=retake_variance,
            add_retake_noise=add_retake_noise,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_latents=src_latents,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_latents=ref_latents,
        )

        return target_latents

    def task_text2music(
        self,
        encoder_text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speaker_embeds: torch.Tensor,
        lyric_token_idx: torch.Tensor,
        lyric_mask: torch.Tensor,
        random_generators: list,
        encoder_text_hidden_states_null: torch.Tensor | None,
        retake_random_generators: list | None,
        latents: torch.Tensor | None = None,
        audio_duration: float = 60.0,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str | None = None,
        retake_variance: float = 0.5,
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str | None = None,
        replace_embeds: bool = False,
        replace_embeds_params: dict = {},
    ):
        add_retake_noise = False
        src_latents = None
        assert src_audio_path is None, (
            "src_audio_path is not supported for text2music task"
        )

        assert ref_audio_input is None and audio2audio_enable is False, (
            "ref_audio_input and audio2audio_enable are not supported for text2music task"
        )
        ref_latents = None

        target_latents = self.text2music_diffusion_process(
            duration=audio_duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            infer_steps=infer_step,
            random_generators=random_generators,
            latents=latents,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            oss_steps=oss_steps,
            encoder_text_hidden_states_null=encoder_text_hidden_states_null,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            retake_random_generators=retake_random_generators,
            retake_variance=retake_variance,
            add_retake_noise=add_retake_noise,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_latents=src_latents,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_latents=ref_latents,
            replace_embeds=replace_embeds,
            replace_embeds_params=replace_embeds_params,
        )
        return target_latents

    @cpu_offload("music_dcae")
    def decode_latents_to_audios(
        self,
        latents,
        target_wav_duration_second=30,
        sample_rate=SAMPLE_RATE,
    ):
        pred_latents = latents
        with torch.no_grad():
            if self.overlapped_decode and target_wav_duration_second > 48:
                _, pred_wavs = self.music_dcae.decode_overlap(
                    pred_latents, sr=sample_rate
                )
            else:
                _, pred_wavs = self.music_dcae.decode(pred_latents, sr=sample_rate)
        pred_wavs = torch.stack([pred_wav.cpu().float() for pred_wav in pred_wavs])

        return pred_wavs

    def save_audios(self, audios, save_path=None, sample_rate=SAMPLE_RATE, fmt="wav"):
        output_audio_paths = []
        bs = audios.shape[0]

        for i in range(bs):
            output_audio_path = self.save_wav_file(
                audios[i],
                i,
                save_path=save_path,
                sample_rate=sample_rate,
                format=fmt,
            )
            output_audio_paths.append(output_audio_path)
        return output_audio_paths

    def prepare_latents(self, batch_size: int, audio_duration: float, seed: int):
        import math

        generator = torch.Generator(device="cpu").manual_seed(seed)
        frame_length = math.ceil(audio_duration * 44100 / 512 / 8)
        # print(f"prepare_latents: {frame_length=}")

        latents = randn_tensor(
            shape=(batch_size, 8, 16, frame_length),
            generator=generator,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        return latents

    def generate(
        self,
        prompt: str | list[str],
        audio_duration: float = 60.0,
        lyrics: str | list[str] = "",
        infer_step: int = 50,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seed: int = 42,
        latents: torch.Tensor = None,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 0.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        return_type: str = "audio",
    ):
        if latents is not None:
            audio_duration = (latents.shape[-1] / 44100) * 512 * 8
        assert 0 < audio_duration <= 240.0, (
            "audio_duration must be between 0 and 240 seconds"
        )
        assert return_type in {"latent", "audio"}, (
            "Invalid return_type, must be one of: latent, audio, path"
        )

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(lyrics, str):
            lyrics = [lyrics]
        if len(prompt) > 1 and len(lyrics) == 1:
            lyrics = lyrics * len(prompt)
        assert len(prompt) == len(lyrics), "prompt and lyrics must have the same length"
        batch_size = len(prompt)

        random_generators, _ = self.set_seeds(batch_size, manual_seed)
        if latents is None:
            latents = self.prepare_latents(batch_size, audio_duration, manual_seed)

        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            prompt
        )

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(prompt)

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # 6 lyric
        lyric_token_ids = []
        lyric_masks = []
        for lyric in lyrics:
            if len(lyric) > 0:
                lyric_token_idx = self.tokenize_lyrics(lyric, debug=False)
                lyric_mask = [1] * len(lyric_token_idx)
            else:
                lyric_token_idx = [0]
                lyric_mask = [0]
            lyric_token_ids.append(lyric_token_idx)
            lyric_masks.append(lyric_mask)

        # Pad all sequences to the same length (max length in batch)
        max_len = max(len(seq) for seq in lyric_token_ids)
        padded_token_ids = []
        padded_masks = []
        for token_ids, mask in zip(lyric_token_ids, lyric_masks):
            pad_len = max_len - len(token_ids)
            padded_token_ids.append(token_ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        lyric_token_ids = torch.tensor(
            padded_token_ids, device=self.device, dtype=torch.long
        )
        lyric_masks = torch.tensor(padded_masks, device=self.device, dtype=torch.long)

        kwargs = {
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "speaker_embeds": speaker_embeds,
            "lyric_token_idx": lyric_token_ids,
            "lyric_mask": lyric_masks,
            "random_generators": random_generators,
            "latents": latents,
            "encoder_text_hidden_states_null": encoder_text_hidden_states_null,
            "retake_random_generators": None,
            "audio_duration": audio_duration,
            "oss_steps": [],
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "omega_scale": omega_scale,
            "guidance_interval": guidance_interval,
            "guidance_interval_decay": guidance_interval_decay,
            "min_guidance_scale": min_guidance_scale,
            "use_erg_lyric": use_erg_lyric,
            "use_erg_diffusion": use_erg_diffusion,
            "guidance_scale_text": guidance_scale_text,
            "guidance_scale_lyric": guidance_scale_lyric,
        }
        target_latents = self.task_text2music(**kwargs)

        if return_type == "latent":
            output = target_latents.cpu()
        else:
            output = self.decode_latents_to_audios(
                latents=target_latents,
                target_wav_duration_second=audio_duration,
                sample_rate=SAMPLE_RATE,
            )
        self.cleanup_memory()

        return output

    def edit_audio_flowedit(
        self,
        source_prompt: str = None,
        source_lyrics: str = "",
        target_prompt: str = None,
        target_lyrics: str = "",
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        infer_step: int = 60,
        guidance_scale: float = 10.0,
        scheduler_type: str = "euler",
        manual_seeds: list | str | int | None = None,
        use_erg_tag: bool = False,  # True?
        oss_steps: str = None,
        src_audio_path: str = None,
        save_path: str = None,
        batch_size: int = 1,
        cfg_type: str = "apg",
        return_type: str = "audio",
        layers_to_hook: list[str] | None = None,
    ):
        assert return_type in {"latent", "audio", "path"}, (
            "Invalid return_type, must be one of: latent, audio, path"
        )
        # read audio_duration from audio file
        audio_duration = self._get_audio_duration(src_audio_path)
        assert 0 < audio_duration <= 240.0, (
            "audio_duration must be between 0 and 240 seconds"
        )

        retake_random_generators, _ = self.set_seeds(batch_size, manual_seeds)
        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = [source_prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts
        )
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(
                batch_size, 1, 1
            )

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(source_lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(source_lyrics, debug=False)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = (
                torch.tensor(lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        kwargs = {
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "speaker_embeds": speaker_embeds,
            "lyric_token_idx": lyric_token_idx,
            "lyric_mask": lyric_mask,
            "retake_random_generators": retake_random_generators,
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "src_audio_path": src_audio_path,
            "edit_target_prompt": target_prompt,
            "edit_target_lyrics": target_lyrics,
            "edit_n_min": edit_n_min,
            "edit_n_max": edit_n_max,
            "edit_n_avg": edit_n_avg,
            "batch_size": batch_size,
            "layers_to_hook": layers_to_hook,
            "cfg_type": cfg_type,
        }
        target_latents = self.task_edit_flowedit(**kwargs)

        if return_type == "latent":
            output = target_latents.cpu()
        else:
            output = self.decode_latents_to_audios(
                latents=target_latents,
                target_wav_duration_second=audio_duration,
                sample_rate=SAMPLE_RATE,
            )
            if return_type == "path":
                output = self.save_audios(
                    audios=output,
                    save_path=save_path,
                    sample_rate=SAMPLE_RATE,
                    fmt="wav",
                )
        self.cleanup_memory()

        return output

    def edit_audio(
        self,
        source_prompt: str = None,
        source_lyrics: str = "",
        target_prompt: str = None,
        target_lyrics: str = "",
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        infer_step: int = 60,
        guidance_scale: float = 10.0,
        scheduler_type: str = "euler",
        manual_seeds: list | str | int | None = None,
        use_erg_tag: bool = False,  # True?
        oss_steps: str = None,
        src_audio_path: str = None,
        save_path: str = None,
        batch_size: int = 1,
        return_type: str = "audio",
    ):
        assert return_type in {"latent", "audio", "path"}, (
            "Invalid return_type, must be one of: latent, audio, path"
        )
        # read audio_duration from audio file
        audio_duration = self._get_audio_duration(src_audio_path)
        assert 0 < audio_duration <= 240.0, (
            "audio_duration must be between 0 and 240 seconds"
        )

        retake_random_generators, _ = self.set_seeds(batch_size, manual_seeds)
        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = [source_prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts
        )
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(
                batch_size, 1, 1
            )

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(source_lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(source_lyrics, debug=False)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = (
                torch.tensor(lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        kwargs = {
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "speaker_embeds": speaker_embeds,
            "lyric_token_idx": lyric_token_idx,
            "lyric_mask": lyric_mask,
            "retake_random_generators": retake_random_generators,
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "src_audio_path": src_audio_path,
            "edit_target_prompt": target_prompt,
            "edit_target_lyrics": target_lyrics,
            "edit_n_min": edit_n_min,
            "edit_n_max": edit_n_max,
            "edit_n_avg": edit_n_avg,
            "batch_size": batch_size,
        }
        target_latents = self.task_edit(**kwargs)

        if return_type == "latent":
            output = target_latents.cpu()
        else:
            output = self.decode_latents_to_audios(
                latents=target_latents,
                target_wav_duration_second=audio_duration,
                sample_rate=SAMPLE_RATE,
            )
            if return_type == "path":
                output = self.save_audios(
                    audios=output,
                    save_path=save_path,
                    sample_rate=SAMPLE_RATE,
                    fmt="wav",
                )
        self.cleanup_memory()

        return output

    def edit_audio_ode_inversion(
        self,
        source_prompt: str = None,
        source_lyrics: str = "",
        target_prompt: str = None,
        target_lyrics: str = "",
        infer_step: int = 60,
        tstart: int = 60,
        guidance_scale_inversion: float = 1.0,
        guidance_scale_editing: float = 10.0,
        scheduler_type: str = "euler",
        manual_seeds: list | str | int | None = None,
        use_erg_tag: bool = False,  # True?
        oss_steps: str = None,
        src_audio_path: str = None,
        save_path: str = None,
        batch_size: int = 1,
        cfg_type: str = "apg",
        return_type: str = "audio",
        layers_to_hook: list[str] | None = None,
    ):
        assert return_type in {
            "latent",
            "audio",
            "path",
            "all_latents",
        }, "Invalid return_type, must be one of: latent, audio, path, all_latents"
        # read audio_duration from audio file
        audio_duration = self._get_audio_duration(src_audio_path)
        assert 0 < audio_duration <= 240.0, (
            "audio_duration must be between 0 and 240 seconds"
        )

        retake_random_generators, _ = self.set_seeds(batch_size, manual_seeds)
        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = [source_prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts
        )
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(
                batch_size, 1, 1
            )

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(source_lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(source_lyrics, debug=False)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = (
                torch.tensor(lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        kwargs = {
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "speaker_embeds": speaker_embeds,
            "lyric_token_idx": lyric_token_idx,
            "lyric_mask": lyric_mask,
            "retake_random_generators": retake_random_generators,
            "infer_step": infer_step,
            "tstart": tstart,
            "guidance_scale_inversion": guidance_scale_inversion,
            "guidance_scale_editing": guidance_scale_editing,
            "scheduler_type": scheduler_type,
            "src_audio_path": src_audio_path,
            "edit_target_prompt": target_prompt,
            "edit_target_lyrics": target_lyrics,
            "batch_size": batch_size,
            "cfg_type": cfg_type,
            "layers_to_hook": layers_to_hook,
        }
        edit_results = self.task_edit_ode_inversion(**kwargs)

        if return_type == "all_latents":
            output = {
                "target_latents": edit_results["target_latents"].cpu(),
                "target_samples": edit_results["target_samples"].cpu(),
                "all_inverted_latents": [
                    latent.cpu() for latent in edit_results["all_inverted_latents"]
                ],
                "all_sampled_latents": [
                    latent.cpu() for latent in edit_results["all_sampled_latents"]
                ],
            }
        elif return_type == "latent":
            target_samples = edit_results["target_samples"]
            output = target_samples.cpu()
        else:
            target_samples = edit_results["target_samples"]
            output = self.decode_latents_to_audios(
                latents=target_samples,
                target_wav_duration_second=audio_duration,
                sample_rate=SAMPLE_RATE,
            )
            if return_type == "path":
                output = self.save_audios(
                    audios=output,
                    save_path=save_path,
                    sample_rate=SAMPLE_RATE,
                    fmt="wav",
                )
        self.cleanup_memory()

        return output

    def __call__(
        self,
        format: str = "wav",
        audio_duration: float = 60.0,
        prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None,
        guidance_interval: float = 1.0,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str = None,
        retake_seeds: list = None,
        retake_variance: float = 0.5,
        task: str = "text2music",
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        save_path: str = None,
        batch_size: int = 1,
        debug: bool = False,
        return_type: str = "path",
    ):
        assert task in {
            "text2music",
            "audio2audio",
            "edit",
            "repaint",
            "extend",
            "retake",
        }, (
            "Invalid task, must be one of: text2music, audio2audio, edit, repaint, extend, retake"
        )
        assert 0 < audio_duration <= 240.0, (
            "audio_duration must be between 0 and 240 seconds"
        )
        assert return_type in {"latent", "audio", "path"}, (
            "Invalid return_type, must be one of: latent, audio, path"
        )

        if audio2audio_enable and ref_audio_input is not None:
            task = "audio2audio"

        random_generators, _ = self.set_seeds(batch_size, manual_seeds)
        retake_random_generators, _ = self.set_seeds(batch_size, retake_seeds)

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = [prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts
        )
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(
                batch_size, 1, 1
            )

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # 6 lyric
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(lyrics, debug=debug)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = (
                torch.tensor(lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        if task == "edit":
            call_function = self.task_edit
            kwargs = {
                "encoder_text_hidden_states": encoder_text_hidden_states,
                "text_attention_mask": text_attention_mask,
                "speaker_embeds": speaker_embeds,
                "lyric_token_idx": lyric_token_idx,
                "lyric_mask": lyric_mask,
                "retake_random_generators": retake_random_generators,
                "infer_step": infer_step,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "src_audio_path": src_audio_path,
                "edit_target_prompt": edit_target_prompt,
                "edit_target_lyrics": edit_target_lyrics,
                "edit_n_min": edit_n_min,
                "edit_n_max": edit_n_max,
                "edit_n_avg": edit_n_avg,
                "batch_size": batch_size,
            }
        else:
            kwargs = {
                "encoder_text_hidden_states": encoder_text_hidden_states,
                "text_attention_mask": text_attention_mask,
                "speaker_embeds": speaker_embeds,
                "lyric_token_idx": lyric_token_idx,
                "lyric_mask": lyric_mask,
                "random_generators": random_generators,
                "encoder_text_hidden_states_null": encoder_text_hidden_states_null,
                "retake_random_generators": retake_random_generators,
                "audio_duration": audio_duration,
                "infer_step": infer_step,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "cfg_type": cfg_type,
                "omega_scale": omega_scale,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
                "min_guidance_scale": min_guidance_scale,
                "use_erg_lyric": use_erg_lyric,
                "use_erg_diffusion": use_erg_diffusion,
                "oss_steps": oss_steps,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "audio2audio_enable": audio2audio_enable,
                "ref_audio_strength": ref_audio_strength,
                "retake_variance": retake_variance,
                "repaint_start": repaint_start,
                "repaint_end": repaint_end,
                "src_audio_path": src_audio_path,
            }
            if task == "repaint":
                call_function = self.task_repaint
            elif task == "extend":
                call_function = self.task_extend
            elif task == "retake":
                call_function = self.task_retake
            elif task == "audio2audio":
                call_function = self.task_audio2audio
            elif task == "text2music":
                call_function = self.task_text2music
        target_latents = call_function(**kwargs)

        if return_type == "latent":
            output = target_latents.cpu()
        else:
            output = self.decode_latents_to_audios(
                latents=target_latents,
                target_wav_duration_second=audio_duration,
                sample_rate=SAMPLE_RATE,
            )
            if return_type == "path":
                output = self.save_audios(
                    audios=output,
                    save_path=save_path,
                    sample_rate=SAMPLE_RATE,
                    fmt=format,
                )
        self.cleanup_memory()

        return output
