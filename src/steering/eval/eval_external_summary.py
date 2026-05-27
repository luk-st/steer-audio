"""
Aggregate the per-concept external-validation correlation tables produced by
``eval_external_mood.py``, ``eval_external_tempo.py`` and
``eval_external_vocals.py`` into a single summary CSV.

Each per-concept script writes ``protocol_results/external_validation_<concept>.csv``
under its own steering directory.  This helper just reads all three and
concatenates them with a ``concept`` column so the headline numbers
(Pearson/Spearman of CLAP/MuQ vs the orthogonal acoustic metric) live in
one place.

Usage:
    python steering/eval/eval_external_summary.py \
        --mood_dir   pwr-mount/steering/outputs/ace_step/concept_mood/loc/sae \
        --tempo_dir  pwr-mount/steering/outputs/ace_step/concept_tempo/loc/sae \
        --vocals_dir pwr-mount/steering/outputs/ace_step/concept_vocal_gender/loc/sae \
        --output     pwr-mount/steering/outputs/ace_step/external_validation_summary.csv
"""

import os
from typing import Optional

import pandas as pd
from fire import Fire


def _load(steering_dir: str, concept: str) -> Optional[pd.DataFrame]:
    if steering_dir is None:
        return None
    path = os.path.join(steering_dir, "protocol_results", f"external_validation_{concept}.csv")
    if not os.path.exists(path):
        print(f"[warn] missing {path}; skipping {concept}")
        return None
    df = pd.read_csv(path)
    df.insert(0, "concept", concept)
    return df


def main(
    mood_dir: Optional[str] = None,
    tempo_dir: Optional[str] = None,
    vocals_dir: Optional[str] = None,
    output: str = "external_validation_summary.csv",
):
    parts = [
        _load(mood_dir, "mood"),
        _load(tempo_dir, "tempo"),
        _load(vocals_dir, "vocals"),
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise SystemExit("No per-concept tables found; run the per-concept scripts first.")
    summary = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved {output} ({len(summary)} rows)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    Fire(main)
