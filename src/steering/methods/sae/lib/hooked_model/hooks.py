from pickle import FALSE
import einops
import numpy as np
import torch


class SAEReconstructHook:
    def __init__(
        self,
        sae,
    ):
        self.sae = sae

    @torch.no_grad()
    def __call__(self, module, input, output):
        output1, output2 = output[0].chunk(2)
        # reshape to SAE input shape
        output1 = output1.permute(0, 2, 3, 1).reshape(len(output1), output1.shape[-1] * output1.shape[-2], -1)
        output2 = output2.permute(0, 2, 3, 1).reshape(len(output2), output2.shape[-1] * output2.shape[-2], -1)
        output_cat = torch.cat([output1, output2], dim=0)
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec
        sae_out1 = sae_out[: output1.shape[1] * len(output1)]
        sae_out2 = sae_out[output1.shape[1] * len(output1) :]
        hook_output = torch.cat(
            [
                sae_out1.reshape(
                    len(output1),
                    int(np.sqrt(output1.shape[-2])),
                    int(np.sqrt(output1.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
                sae_out2.reshape(
                    len(output2),
                    int(np.sqrt(output2.shape[-2])),
                    int(np.sqrt(output2.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
            ],
            dim=0,
        )

        return (hook_output,)


class AblateHook:
    @torch.no_grad()
    def __call__(self, module, input, output):
        # if isinstance(input, tuple):
        #     return input[0]
        # return input[0]
        # return torch.zeros_like(input[0])
        print(len(input))
        print(input)
        return input


class StableAudioSAEReconstructHook:
    def __init__(self, sae, along_freqs=False, both_preds=False):
        self.sae = sae
        self.along_freqs = along_freqs
        self.both_preds = both_preds

    @torch.no_grad()
    def __call__(self, module, input, output):
        if self.both_preds:
            to_intervene = output
        else:
            output1, output2 = output.chunk(2)
            to_intervene = output2
        batch_size = to_intervene.shape[0]
        if self.along_freqs:
            to_intervene = einops.rearrange(to_intervene, "b t f -> b f t")
        sae_input, _, _ = self.sae.preprocess_input(to_intervene)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec
        if self.along_freqs:
            sae_out = einops.rearrange(sae_out, "(b f) t -> b t f", b=batch_size)
        else:
            sae_out = einops.rearrange(sae_out, "(b t) f -> b t f", b=batch_size)
        if self.both_preds:
            hook_output = sae_out
        else:
            hook_output = torch.cat(
                [output1, sae_out],
                dim=0,
            )

        return hook_output


class StableAudioInterventionHook:
    def __init__(
        self,
        sae,
        latent_idx: int | list[int],
        multiplier: float | list[float],
        along_freqs=False,
        both_preds=False,
    ):
        """Simple hook that multiplies specific SAE feature activations by a scalar.

        Args:
            sae: The trained SAE model
            feature_idx: Index of the feature to modify
            multiplier: Factor to multiply the feature activation by
        """
        self.sae = sae
        self.along_freqs = along_freqs
        self.both_preds = both_preds

        if isinstance(latent_idx, int):
            latent_idx = [latent_idx]
        if isinstance(multiplier, float):
            multiplier = [multiplier]
        if len(multiplier) == 1:
            multiplier = multiplier * len(latent_idx)
        assert len(latent_idx) == len(multiplier), "Length of latent_idx and multiplier must be the same"
        self.latent_idx = latent_idx
        self.multiplier = multiplier

    @torch.no_grad()
    def __call__(self, module, input, output):
        if self.both_preds:
            to_intervene = output
        else:
            output1, output2 = output.chunk(2)
            to_intervene = output2
        batch_size = to_intervene.shape[0]
        if self.along_freqs:
            to_intervene = einops.rearrange(to_intervene, "b t f -> b f t")
        sae_input, _, _ = self.sae.preprocess_input(to_intervene)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)

        # modify latent and decode
        for latent_idx, multiplier in zip(self.latent_idx, self.multiplier):
            latents[..., latent_idx] *= multiplier
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec

        if self.along_freqs:
            sae_out = einops.rearrange(sae_out, "(b f) t -> b t f", b=batch_size)
        else:
            sae_out = einops.rearrange(sae_out, "(b t) f -> b t f", b=batch_size)
        if self.both_preds:
            hook_output = sae_out
        else:
            hook_output = torch.cat(
                [output1, sae_out],
                dim=0,
            )

        return hook_output


class StableAudioAblateHook:
    def __init__(
        self,
        sae,
        along_freqs=False,
        both_preds=False,
    ):
        """Simple hook that multiplies specific SAE feature activations by a scalar.

        Args:
            sae: The trained SAE model
            along_freqs: Whether to operate along frequency axis
        """
        self.sae = sae
        self.along_freqs = along_freqs
        self.both_preds = both_preds

    @torch.no_grad()
    def __call__(self, module, input, output):
        if self.both_preds:
            to_intervene = output
        else:
            output1, output2 = output.chunk(2)
            to_intervene = output2
        batch_size = to_intervene.shape[0]
        if self.along_freqs:
            to_intervene = einops.rearrange(to_intervene, "b t f -> b f t")
        # inside preprocess_input: x = x.reshape(batch_size * sample_size, emb_size)
        sae_input, _, _ = self.sae.preprocess_input(to_intervene)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)

        # zero out latent and decode
        # latents[..., :] = 0
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec
        sae_out = torch.zeros_like(sae_out)

        if self.along_freqs:
            sae_out = einops.rearrange(sae_out, "(b f) t -> b t f", b=batch_size)
        else:
            sae_out = einops.rearrange(sae_out, "(b t) f -> b t f", b=batch_size)
        if self.both_preds:
            hook_output = sae_out
        else:
            hook_output = torch.cat(
                [output1, sae_out],
                dim=0,
        )

        return hook_output
