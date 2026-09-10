import abc
from collections import defaultdict
from typing import Tuple, Union

import torch


def compute_num_cfg_passes(guidance_scale_text=0.0, guidance_scale_lyric=0.0):
    """
    Compute number of CFG passes based on guidance parameters.

    ACE-Step uses:
    - 2 passes (default): cond, uncond
    - 3 passes (double guidance): cond, cond_text_only, uncond

    Args:
        guidance_scale_text: Text guidance scale
        guidance_scale_lyric: Lyric guidance scale

    Returns:
        Number of CFG passes (2 or 3)
    """
    do_double_condition_guidance = (
        guidance_scale_text is not None
        and guidance_scale_text > 1.0
        and guidance_scale_lyric is not None
        and guidance_scale_lyric > 1.0
    )
    return 3 if do_double_condition_guidance else 2


class VectorControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_ace: str):
        raise NotImplementedError

    def __call__(self, vector, place_in_ace: str):
        vector = self.forward(vector, place_in_ace)  # type: ignore
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
            self.cur_step += 1
        return vector


class VectorStore(VectorControl):
    def __init__(
        self,
        steering_vectors=None,
        steer=True,
        alpha=10,
        beta=2,
        steer_back=False,
        device="cpu",
        save_only_cond=True,
        steer_mode="cond_only",
        num_cfg_passes=None,
        renorm_after_steer=False,
        start_step=0,
    ):
        """
        Args:
            steering_vectors: Pre-computed steering vectors (dict)
            steer: Whether to apply steering
            alpha: Steering strength multiplier
            beta: Backward steering strength (for steer_back=True)
            steer_back: If True, remove concept; if False, add concept
            device: Device for tensors
            save_only_cond: If True, only save cond activations when computing vectors
            renorm_after_steer: If True, renormalize the steered activation back to its
                original L2 norm (matches CAA paper). Default False — discarded after
                ablation showed it's redundant for ACE-Step (RMSNorm downstream) and
                actively harmful for AudioLDM2 (compresses dynamic range).
            steer_mode: How to apply steering vectors. Options:
                - 'cond_only': Steer only cond using cond vectors (steering scales with CFG)
                - 'uncond_only': Steer only uncond using uncond vectors (inverse CFG scaling)
                - 'uncond_for_cond': Steer only cond using uncond vectors (steering scales with CFG)
                - 'separate': Steer cond with cond vectors, uncond with uncond vectors (independent)
                - 'both_cond': Steer both cond and uncond using cond vectors (CFG-independent)
                - 'both_uncond': Steer both cond and uncond using uncond vectors (CFG-independent)
            num_cfg_passes: Number of CFG passes (2 or 3), None for auto-detect
        """
        super(VectorStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.steering_vectors = steering_vectors
        self.steer = steer
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.device = device

        # Track CFG passes in ACE-Step
        # 2 passes (default): cond, uncond
        # 3 passes (with double guidance): cond, cond_text_only, uncond
        self.num_cfg_passes = num_cfg_passes  # If None, auto-detect from forward passes
        self.cfg_pass_count = 0  # Tracks which CFG pass we're in (0=cond, 1=text_only or uncond, 2=uncond)
        self.save_only_cond = save_only_cond  # If True, only save activations from conditional pass

        # Steering mode validation and setup
        valid_modes = [
            "cond_only",
            "uncond_only",
            "uncond_for_cond",
            "separate",
            "both_cond",
            "both_uncond",
        ]
        if steer_mode not in valid_modes:
            raise ValueError(f"steer_mode must be one of {valid_modes}, got {steer_mode}")
        self.steer_mode = steer_mode
        self.renorm_after_steer = renorm_after_steer
        self.start_step = int(start_step)

        self.actual_denoising_step = 0  # Tracks actual denoising steps (not CFG passes)

    def reset(self):
        super(VectorStore, self).reset()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.cfg_pass_count = 0
        self.actual_denoising_step = 0

    @staticmethod
    def get_empty_store():
        return defaultdict(list)

    def forward(self, vector, place_in_ace: str):  # type: ignore
        # Determine if and how to steer based on steering mode
        should_steer = False
        # Skip steering for the first `start_step` diffusion steps.
        if self.steer and self.actual_denoising_step >= self.start_step:
            if self.steer_mode == "cond_only":
                # Only steer conditional pass (cfg_pass_count == 0)
                should_steer = self.cfg_pass_count == 0
            elif self.steer_mode == "uncond_only":
                # Only steer unconditional pass (last CFG pass)
                # For 2-pass: uncond is pass 1, for 3-pass: uncond is pass 2
                if self.num_cfg_passes is not None:
                    should_steer = self.cfg_pass_count == (self.num_cfg_passes - 1)
                else:
                    # Default to 2-pass mode if num_cfg_passes not specified
                    should_steer = self.cfg_pass_count == 1
            elif self.steer_mode == "uncond_for_cond":
                # Steer only conditional pass using uncond vectors
                should_steer = self.cfg_pass_count == 0
            elif self.steer_mode == "separate":
                # Steer all passes, but with different vectors per pass
                should_steer = True
            elif self.steer_mode == "both_cond":
                # Steer all passes with the same cond vector
                should_steer = True
            elif self.steer_mode == "both_uncond":
                # Steer all passes with the same uncond vector
                should_steer = True

        if should_steer:
            # Determine steering vector key based on format and mode
            if not self.steering_vectors:
                raise ValueError("Cannot steer: steering_vectors is empty")
            first_key = list(self.steering_vectors.keys())[0]

            # Determine which steering vector to use
            if isinstance(first_key, tuple):
                # Steering vectors stored per CFG pass: key = (denoising_step, cfg_pass)
                if len(self.steering_vectors) == 1:
                    # Turbo version: single key for all steps
                    steer_key = first_key
                else:
                    # Full version: key per step and CFG pass.
                    # With save_only_cond counting, actual_denoising_step is advanced at
                    # the END of the cond pass, so during the later passes of the same
                    # denoising step (cfg_pass_count > 0) it already points one step
                    # ahead; correct the lookup back to the current step.
                    lookup_step = self.actual_denoising_step
                    if self.save_only_cond and self.cfg_pass_count > 0:
                        lookup_step -= 1
                    if self.steer_mode == "cond_only" or self.steer_mode == "both_cond":
                        # Use conditional pass vectors (cfg_pass=0) for steering
                        steer_key = (lookup_step, 0)
                    elif (
                        self.steer_mode == "uncond_only"
                        or self.steer_mode == "uncond_for_cond"
                        or self.steer_mode == "both_uncond"
                    ):
                        # Use unconditional pass vectors (last cfg_pass)
                        uncond_pass = (self.num_cfg_passes - 1) if self.num_cfg_passes is not None else 1
                        steer_key = (lookup_step, uncond_pass)
                    elif self.steer_mode == "separate":
                        # Use vectors from current CFG pass
                        steer_key = (lookup_step, self.cfg_pass_count)
            else:
                # Steering vectors stored per denoising step only: key = denoising_step
                # These were computed with save_only_cond=True (only conditional pass)
                if self.steer_mode == "separate":
                    raise ValueError(
                        "Cannot use steer_mode='separate' with steering vectors "
                        "that were computed only for conditional pass (save_only_cond=True). "
                        "Recompute steering vectors with save_only_cond=False."
                    )
                if self.steer_mode == "uncond_only":
                    raise ValueError(
                        "Cannot use steer_mode='uncond_only' with steering vectors "
                        "that were computed only for conditional pass (save_only_cond=True). "
                        "Recompute steering vectors with save_only_cond=False."
                    )
                if self.steer_mode == "uncond_for_cond":
                    raise ValueError(
                        "Cannot use steer_mode='uncond_for_cond' with steering vectors "
                        "that were computed only for conditional pass (save_only_cond=True). "
                        "Recompute steering vectors with save_only_cond=False."
                    )
                if self.steer_mode == "both_uncond":
                    raise ValueError(
                        "Cannot use steer_mode='both_uncond' with steering vectors "
                        "that were computed only for conditional pass (save_only_cond=True). "
                        "Recompute steering vectors with save_only_cond=False."
                    )
                if len(self.steering_vectors) == 1:
                    # Turbo version: single key for all steps
                    steer_key = first_key
                else:
                    # Full version: key per step
                    steer_key = self.actual_denoising_step

            steering_vector = self.steering_vectors[steer_key][place_in_ace][len(self.step_store[place_in_ace])]  # type: ignore
            # Convert to tensor with same dtype as the input vector
            steering_vector = torch.tensor(steering_vector, dtype=vector.dtype, device=self.device).view(1, 1, -1)
            # save current norm of vector components
            norm = torch.norm(vector, dim=2, keepdim=True)

            # Skip all processing if alpha is 0 (no steering)
            if self.alpha == 0 and not self.steer_back:
                pass  # Return vector unchanged
            elif self.steer_back:
                # steering backward, i.e. removing notion from vector

                # computing dot products between vector components and steering vector x
                sim = torch.tensordot(vector, steering_vector, dims=([2], [2])).view(  # type: ignore
                    vector.size()[0], vector.size()[1], 1
                )
                # we will steer back only if dot product is positive, i.e.
                # if there's positive amount of information from steering vector in the vector
                sim = torch.where(sim > 0, sim, 0)

                # steer backward for beta*sim
                vector = vector - (self.beta * sim) * steering_vector.expand(1, vector.size()[1], -1)

                if self.renorm_after_steer:
                    # renormalize so that the norm of the steered vector is the same as of original one
                    vector = vector / torch.norm(vector, dim=2, keepdim=True)
                    vector = vector * norm
            else:
                # steer forward, i.e. add a steering vector x multiplied by self.alpha
                vector = vector + self.alpha * steering_vector.expand(1, vector.size()[1], -1)

                if self.renorm_after_steer:
                    # renormalize so that the norm of the steered vector is the same as of original one
                    vector = vector / torch.norm(vector, dim=2, keepdim=True)
                    vector = vector * norm

        # save activation (vector) for further computing steering vectors
        # ACE-Step uses separate forward passes for CFG (not batched like SDXL)
        # Order: 1) cond, 2) cond_text_only (optional), 3) uncond
        # Only save from conditional pass if save_only_cond is True (recommended)
        should_save = (not self.save_only_cond) or (self.cfg_pass_count == 0)

        if should_save:
            # Convert to float32 first since numpy doesn't support bfloat16
            vector_to_store = vector.data.cpu().float().numpy()
            # No batch splitting needed - ACE does sequential passes, not batched
            self.step_store[place_in_ace].append(vector_to_store.mean(axis=0).mean(axis=0))

        return vector

    def between_steps(self):
        # Check if we saved data
        has_data = self.step_store and any(self.step_store.values())

        if self.save_only_cond:
            # Only saving conditional pass (cfg_pass_count=0)
            if has_data:
                # We have data - this was the conditional pass
                self.vector_store[self.actual_denoising_step] = self.step_store
                self.actual_denoising_step += 1
                # Increment to next CFG pass
                self.cfg_pass_count = 1
            else:
                # No data - this was a non-conditional CFG pass
                self.cfg_pass_count += 1
                # Reset when we reach the expected number of CFG passes
                if self.num_cfg_passes is not None and self.cfg_pass_count >= self.num_cfg_passes:
                    self.cfg_pass_count = 0
        else:
            # Saving all CFG passes - store separately for each pass
            if has_data:
                # Store with key that includes cfg_pass_count
                key = (self.actual_denoising_step, self.cfg_pass_count)
                self.vector_store[key] = self.step_store
                self.cfg_pass_count += 1

                # When we complete all CFG passes, move to next denoising step
                if self.num_cfg_passes is not None and self.cfg_pass_count >= self.num_cfg_passes:
                    self.cfg_pass_count = 0
                    self.actual_denoising_step += 1

        # Always clear step_store for next pass
        self.step_store = self.get_empty_store()


def register_vector_control(model, controller, verbose=False, explicit_layers=None):
    def block_forward(self, place_in_ace: str):
        # overriding src.models.ace_step.ACE.acestep.models.attention.LinearTransformerBlock forward function
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor | None = None,
            attention_mask: torch.FloatTensor | None = None,
            encoder_attention_mask: torch.FloatTensor | None = None,
            rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] | None = None,
            rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] | None = None,
            temb: torch.FloatTensor | None = None,
        ):
            N = hidden_states.shape[0]

            # step 1: AdaLN single
            if self.use_adaln_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + temb.reshape(N, 6, -1)  # type: ignore
                ).chunk(6, dim=1)

            norm_hidden_states = self.norm1(hidden_states)
            if self.use_adaln_single:
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa  # type: ignore

            # step 2: attention
            if not self.add_cross_attention:
                attn_output, encoder_hidden_states = self.attn(
                    hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=rotary_freqs_cis_cross,
                )
            else:
                attn_output, _ = self.attn(
                    hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=None,
                )

            if self.use_adaln_single:
                attn_output = gate_msa * attn_output  # type: ignore
            hidden_states = attn_output + hidden_states

            if self.add_cross_attention:
                attn_output = self.cross_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=rotary_freqs_cis_cross,
                )
                # controller here
                attn_output = controller(attn_output, place_in_ace)
                # controller here
                hidden_states = attn_output + hidden_states

            # step 3: add norm
            norm_hidden_states = self.norm2(hidden_states)
            if self.use_adaln_single:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp  # type: ignore

            # step 4: feed forward
            ff_output = self.ff(norm_hidden_states)
            if self.use_adaln_single:
                ff_output = gate_mlp * ff_output  # type: ignore

            hidden_states = hidden_states + ff_output

            return hidden_states

        return forward

    def register_recr(net_, count, place_in_ace, explicit_layers=None):
        """
        registering controller for all the LinearTransformerBlock in the model
        """
        if net_.__class__.__name__ == "LinearTransformerBlock":
            if explicit_layers is not None and place_in_ace not in explicit_layers:
                return count
            net_.forward = block_forward(net_, place_in_ace)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_ace, explicit_layers)
        return count

    block_count = 0
    sub_nets = model.transformer_blocks.named_children()
    counts = {}
    for net in sub_nets:
        name = "tf" + net[0]
        count_in_block = register_recr(net[1], 0, name, explicit_layers)
        block_count += count_in_block
        counts[name] = count_in_block
    if verbose:
        print(f"Total blocks registered: {block_count}")
        print(f"Blocks per transformer: {counts}")
    controller.num_att_layers = block_count
