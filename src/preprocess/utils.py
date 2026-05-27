from pathlib import Path
from typing import Optional

import pandas as pd


def extract_prompts_csv(
    csv_path: str,
    feature: str | None = None,
    prefix_clean: Optional[str] = None,
    suffix_clean: Optional[str] = None,
    prefix_corrupted: Optional[str] = None,
    suffix_corrupted: Optional[str] = None,
):
    """Load (clean, corrupted) prompt pairs.

    ``csv_path`` accepts either a local CSV file path or a HuggingFace Hub
    dataset repo id (e.g. ``lukasz-staniszewski/patching-music-musiccaps-prompts``).
    Local paths take precedence; if the path doesn't exist on disk, it's treated
    as a Hub repo id and loaded via ``datasets.load_dataset``.
    """
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset

        df = load_dataset(csv_path, split="train").to_pandas()

    df_features = df
    if feature is not None:
        df_features = df_features[df_features["original_feature"].isin([feature])]
    clean_prompts = df_features["clean_prompt"].tolist()
    corrupted_prompts = df_features["corrupted_prompt"].tolist()

    prefix_clean = prefix_clean or ""
    suffix_clean = suffix_clean or ""
    clean_prompts = [
        f"{prefix_clean}{prompt}{suffix_clean}" for prompt in clean_prompts
    ]
    prefix_corrupted = prefix_corrupted or ""
    suffix_corrupted = suffix_corrupted or ""
    corrupted_prompts = [
        f"{prefix_corrupted}{prompt}{suffix_corrupted}" for prompt in corrupted_prompts
    ]

    return clean_prompts, corrupted_prompts
