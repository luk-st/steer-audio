"""Prompt-Injection (PI / PCI) steering library — extracted from the original
``steering/freesliders/run_prompt_injection.py``.

Algorithm: instead of computing steering vectors or interpolating embeddings,
this method hard-switches the conditioning prompt during diffusion at a chosen
``switch_step``. Alpha encodes both *direction* and *strength*:

* ``alpha > 0`` → neutral for first ``N_STEPS - alpha`` steps, positive after.
* ``alpha < 0`` → neutral for first ``N_STEPS - abs(alpha)`` steps, negative after.
* ``alpha = 0`` → neutral throughout (baseline).
"""

from __future__ import annotations

import os
import sys

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "src", "models", "ace_step", "ACE"),
):
    if _p not in sys.path:
        sys.path.append(_p)

from src.models.ace_step.ACE.acestep.cpu_offload import cpu_offload
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline


def build_prompt_triple(test_prompt: str, concept: str) -> tuple[str, str, str]:
    """Return ``(neutral, positive, negative)`` for a given concept."""
    p = test_prompt
    concept_configs = {
        "piano": {
            "positive": f"{p}, with piano",
            "negative": f"{p}",
            "neutral": f"{p}, with instrument",
        },
        "mood": {
            "positive": f"happy song, {p}",
            "negative": f"sad song, {p}",
            "neutral": f"a song, {p}",
        },
        "tempo": {
            "positive": f"fast song, {p}",
            "negative": f"slow song, {p}",
            "neutral": f"a song, {p}",
        },
        "vocal_gender": {
            "positive": f"{p}, with female vocal",
            "negative": f"{p}, with male vocal",
            "neutral": f"{p}, with clean vocal",
        },
        "vocal_style": {
            "positive": f"{p}, with rap vocal",
            "negative": f"{p}, with sing vocal",
            "neutral": f"{p}, with clean vocal",
        },
        "guitar_electronic": {
            "positive": f"{p}, with acoustic guitar",
            "negative": f"{p}, with electric guitar",
            "neutral": f"{p}, with a guitar",
        },
        "violin": {
            "positive": f"{p}, with violin",
            "negative": f"{p}",
            "neutral": f"{p}, with instrument",
        },
        "rock_genre": {
            "positive": f"jazz song, {p}",
            "negative": f"rock song, {p}",
            "neutral": f"a song, {p}",
        },
        "electronic_music": {
            "positive": f"classical song, {p}",
            "negative": f"electronic song, {p}",
            "neutral": f"a song, {p}",
        },
    }
    if concept not in concept_configs:
        raise ValueError(
            f"Unknown concept: {concept}. Available: {list(concept_configs.keys())}"
        )
    c = concept_configs[concept]
    return c["neutral"], c["positive"], c["negative"]


@cpu_offload("text_encoder_model")
def compute_batched_pi_embeddings(
    pipe: SimpleACEStepPipeline,
    neutral_prompts: list[str],
    positive_prompts: list[str],
    negative_prompts: list[str],
    use_erg_tag: bool = True,
    tau: float = 0.01,
    l_min: int = 8,
    l_max: int = 10,
) -> dict:
    """Tokenise and embed prompt triples; returns separate ``positive_hs`` and
    ``negative_hs`` (PI hard-switches between them rather than interpolating)."""
    N = len(neutral_prompts)
    assert len(positive_prompts) == N and len(negative_prompts) == N

    all_prompts: list[str] = []
    for n, p, neg in zip(neutral_prompts, positive_prompts, negative_prompts):
        all_prompts.extend([n, p, neg])

    inputs = pipe.text_tokenizer(
        all_prompts, return_tensors="pt", padding=True, truncation=True,
    )
    inputs = {k: v.to(pipe.device) for k, v in inputs.items()}

    if pipe.text_encoder_model.device != pipe.device:
        pipe.text_encoder_model.to(pipe.device)

    neutral_ids = inputs["input_ids"][0::3]
    neutral_attn = inputs["attention_mask"][0::3]
    pos_ids = inputs["input_ids"][1::3]
    pos_attn = inputs["attention_mask"][1::3]
    neg_ids = inputs["input_ids"][2::3]
    neg_attn = inputs["attention_mask"][2::3]

    with torch.no_grad():
        neutral_hs = pipe.text_encoder_model(
            input_ids=neutral_ids, attention_mask=neutral_attn,
        ).last_hidden_state

    pn_ids = torch.cat([pos_ids, neg_ids], dim=0)
    pn_attn = torch.cat([pos_attn, neg_attn], dim=0)
    with torch.no_grad():
        pn_hs = pipe.text_encoder_model(
            input_ids=pn_ids, attention_mask=pn_attn,
        ).last_hidden_state
    positive_hs = pn_hs[:N]
    negative_hs = pn_hs[N:]

    erg_neutral_hs = None
    if use_erg_tag:
        handlers = []

        def hook(module, input, output):
            output[:] *= tau
            return output

        for i in range(l_min, l_max):
            handlers.append(
                pipe.text_encoder_model.encoder.block[i].layer[0]
                .SelfAttention.q.register_forward_hook(hook)
            )

        with torch.no_grad():
            erg_neutral_hs = pipe.text_encoder_model(
                input_ids=neutral_ids, attention_mask=neutral_attn,
            ).last_hidden_state

        for h in handlers:
            h.remove()

    result = {
        "neutral_hs": neutral_hs,
        "positive_hs": positive_hs,
        "negative_hs": negative_hs,
        "neutral_masks": neutral_attn,
        "positive_masks": pos_attn,
        "negative_masks": neg_attn,
    }
    if erg_neutral_hs is not None:
        result["erg_neutral_hs"] = erg_neutral_hs
    return result
