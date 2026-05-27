"""Token-Embedding (TokE) steering library — extracted from
``steering/freesliders/token_embeddings/{run_toke.py, compute_toke_direction.py}``.

Algorithm: compute a per-concept direction vector once (averaging
``positive_hs[concept_token] - negative_hs[concept_token]`` over a set of steer
prompts). At inference, embed each test prompt with a neutral addon that
contains a generic concept word, locate the token, and add
``alpha * direction`` to its hidden state. The resulting modified embeddings
drive ``pipe.task_text2music(replace_embeds=True)``.
"""

from __future__ import annotations

import os
import sys
from typing import Literal, TypedDict

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


# ---------------------------------------------------------------------------
# Concept config
# ---------------------------------------------------------------------------

# Compute-time: how to build positive/negative pairs from a base prompt, plus
# which token in the positive prompt to extract the direction from.
CONCEPT_DIRECTION_CONFIG: dict[str, dict] = {
    "piano": {
        "positive_fn": lambda p: f"{p}, with piano",
        "negative_fn": lambda p: f"{p}, with instrument",
        "concept_word": "piano",
        "occurrence": "last",
    },
    "mood": {
        "positive_fn": lambda p: f"happy song, {p}",
        "negative_fn": lambda p: f"sad song, {p}",
        "concept_word": "song",
        "occurrence": "first",
    },
    "tempo": {
        "positive_fn": lambda p: f"fast song, {p}",
        "negative_fn": lambda p: f"slow song, {p}",
        "concept_word": "song",
        "occurrence": "first",
    },
    "vocal_gender": {
        "positive_fn": lambda p: f"{p}, with female vocal",
        "negative_fn": lambda p: f"{p}, with male vocal",
        "concept_word": "vocal",
        "occurrence": "last",
    },
    "vocal_style": {
        "positive_fn": lambda p: f"{p}, with rap vocal",
        "negative_fn": lambda p: f"{p}, with sing vocal",
        "concept_word": "vocal",
        "occurrence": "last",
    },
    "guitar_electronic": {
        "positive_fn": lambda p: f"{p}, with acoustic guitar",
        "negative_fn": lambda p: f"{p}, with electric guitar",
        "concept_word": "guitar",
        "occurrence": "last",
    },
    "violin": {
        "positive_fn": lambda p: f"{p}, with violin",
        "negative_fn": lambda p: f"{p}, with instrument",
        "concept_word": "violin",
        "occurrence": "last",
    },
    "rock_genre": {
        "positive_fn": lambda p: f"jazz song, {p}",
        "negative_fn": lambda p: f"rock song, {p}",
        "concept_word": "song",
        "occurrence": "first",
    },
    "electronic_music": {
        "positive_fn": lambda p: f"classical song, {p}",
        "negative_fn": lambda p: f"electronic song, {p}",
        "concept_word": "song",
        "occurrence": "first",
    },
}

# Eval-time: which generic word to look for in the (already-prefixed) neutral
# prompt. Pairs with ``CONCEPT_TO_NEUTRAL_ADDON`` from
# ``steering.caa.utils.constants``.
CONCEPT_NEUTRAL_CONFIG: dict[str, dict] = {
    "piano": {"concept_word": "instrument", "occurrence": "last"},
    "mood": {"concept_word": "song", "occurrence": "first"},
    "tempo": {"concept_word": "song", "occurrence": "first"},
    "vocal_gender": {"concept_word": "vocal", "occurrence": "last"},
    "vocal_style": {"concept_word": "vocal", "occurrence": "last"},
    "guitar_electronic": {"concept_word": "guitar", "occurrence": "last"},
    "violin": {"concept_word": "instrument", "occurrence": "last"},
    "rock_genre": {"concept_word": "song", "occurrence": "first"},
    "electronic_music": {"concept_word": "song", "occurrence": "first"},
}


# ---------------------------------------------------------------------------
# Token-search helpers
# ---------------------------------------------------------------------------


class TokenMatch(TypedDict):
    prompt_index: int
    start: int
    end: int


def _find_subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not subsequence:
        return []
    w = len(subsequence)
    return [i for i in range(len(sequence) - w + 1) if sequence[i : i + w] == subsequence]


def find_concept_token_positions(
    *,
    word: str,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    occurrence: Literal["first", "last"] = "last",
) -> list[tuple[int, int]]:
    """Return the (start, end) token span of ``word`` in every row."""
    target_ids: list[int] = tokenizer(text=word, add_special_tokens=False)["input_ids"]
    results: list[tuple[int, int]] = []
    for b in range(input_ids.shape[0]):
        valid_len = int(attention_mask[b].sum().item())
        ids_row = input_ids[b, :valid_len].tolist()
        starts = _find_subsequence_positions(ids_row, target_ids)
        if occurrence == "first" and starts:
            starts = [starts[0]]
        elif occurrence == "last" and starts:
            starts = [starts[-1]]
        assert starts, f"Concept word {word!r} not found in prompt {b}"
        results.append((starts[0], starts[0] + len(target_ids)))
    return results


# ---------------------------------------------------------------------------
# Direction computation (offline)
# ---------------------------------------------------------------------------


def _get_steer_base_prompts(concept: str) -> list[str]:
    """Recover bare base prompts by reversing the concept-specific addon
    encoded in :mod:`steering.sae.lib.configs.steer_prompts`."""
    from src.steering.methods.sae.lib.configs.steer_prompts import CONCEPT_TO_PROMPTS

    if concept not in CONCEPT_TO_PROMPTS:
        raise ValueError(f"Unknown concept: {concept}. Available: {list(CONCEPT_TO_PROMPTS.keys())}")

    neutral, positive, _ = CONCEPT_TO_PROMPTS[concept]()

    if concept in ("piano", "drums", "violin"):
        return list(neutral)
    if concept == "mood":
        return [p[len("happy song, ") :] for p in positive]
    if concept == "tempo":
        return [p[len("fast song, ") :] for p in positive]
    if concept == "vocal_gender":
        return [p[: -len(", with female vocal")] for p in positive]
    if concept == "vocal_style":
        return [p[: -len(", with rap vocal")] for p in positive]
    if concept == "guitar_electronic":
        return [p[: -len(", with acoustic guitar")] for p in positive]
    if concept == "rock_genre":
        return [p[len("jazz song, ") :] for p in positive]
    if concept == "electronic_music":
        return [p[len("classical song, ") :] for p in positive]

    raise ValueError(f"Unsupported concept for TokE direction: {concept}")


@cpu_offload("text_encoder_model")
def compute_direction(pipe: SimpleACEStepPipeline, concept: str) -> torch.Tensor:
    """Compute the averaged TokE direction vector.

    For each (positive, negative) base-prompt pair: locate the concept token in
    the positive prompt, subtract the same-position vector in the negative
    prompt, then average across all pairs. Returns a tensor of shape
    ``[hidden_dim]``.
    """
    cfg = CONCEPT_DIRECTION_CONFIG[concept]
    base_prompts = _get_steer_base_prompts(concept)

    positives = [cfg["positive_fn"](p) for p in base_prompts]
    negatives = [cfg["negative_fn"](p) for p in base_prompts]
    N = len(base_prompts)
    print(f"  {N} steer prompts for concept '{concept}'")

    all_prompts: list[str] = []
    for p, n in zip(positives, negatives):
        all_prompts.extend([p, n])

    inputs = pipe.text_tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(pipe.device) for k, v in inputs.items()}

    if pipe.text_encoder_model.device != pipe.device:
        pipe.text_encoder_model.to(pipe.device)

    with torch.no_grad():
        all_hs = pipe.text_encoder_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state

    positive_hs = all_hs[0::2]
    negative_hs = all_hs[1::2]

    matches = find_concept_token_positions(
        word=cfg["concept_word"],
        tokenizer=pipe.text_tokenizer,
        input_ids=inputs["input_ids"][0::2],
        attention_mask=inputs["attention_mask"][0::2],
        occurrence=cfg["occurrence"],
    )
    assert len(matches) == N

    directions = []
    for i, (s, e) in enumerate(matches):
        assert e == s + 1, f"Expected single token, got span {(e - s)}"
        directions.append(
            positive_hs[i, s:e, :].mean(dim=0) - negative_hs[i, s:e, :].mean(dim=0)
        )
    direction = torch.stack(directions).mean(dim=0)

    per_norm = torch.stack(directions).norm(dim=-1)
    print(
        f"  per-prompt direction norms: mean={per_norm.mean():.4f}, std={per_norm.std():.4f}"
    )
    print(f"  averaged direction norm: {direction.norm():.4f}")
    return direction


# ---------------------------------------------------------------------------
# Neutral-prompt embedding (inference time)
# ---------------------------------------------------------------------------


@cpu_offload("text_encoder_model")
def compute_batched_neutral_embeddings(
    pipe: SimpleACEStepPipeline,
    neutral_prompts: list[str],
    concept_word: str,
    occurrence: Literal["first", "last"],
    use_erg_tag: bool = True,
    tau: float = 0.01,
    l_min: int = 8,
    l_max: int = 10,
) -> dict:
    """Embed neutral prompts and find concept-token positions.

    Returns dict with ``neutral_hs``, ``neutral_masks``, ``seq_ids_in_prompts``
    (concept-token start index per prompt), optionally ``erg_neutral_hs``.
    """
    inputs = pipe.text_tokenizer(
        neutral_prompts, return_tensors="pt", padding=True, truncation=True,
    )
    inputs = {k: v.to(pipe.device) for k, v in inputs.items()}

    if pipe.text_encoder_model.device != pipe.device:
        pipe.text_encoder_model.to(pipe.device)

    with torch.no_grad():
        neutral_hs = pipe.text_encoder_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state

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
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state

        for h in handlers:
            h.remove()

    spans = find_concept_token_positions(
        word=concept_word,
        tokenizer=pipe.text_tokenizer,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        occurrence=occurrence,
    )
    seq_ids = torch.tensor([s for s, _e in spans], device=pipe.device)

    result = {
        "neutral_hs": neutral_hs,
        "neutral_masks": inputs["attention_mask"],
        "seq_ids_in_prompts": seq_ids,
    }
    if erg_neutral_hs is not None:
        result["erg_neutral_hs"] = erg_neutral_hs
    return result
