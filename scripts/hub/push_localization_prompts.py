"""Push the MusicCaps-derived (clean, corrupted) localization-prompt CSV to the
HuggingFace Hub as a :class:`datasets.Dataset`.

The CSV is produced by ``src/preprocess/prepare_prompts.py`` from
``data/music_caps.csv`` by sweeping every feature in
``MUSICCAPS_ORIGINAL_FEATURES`` (see ``src/preprocess/features.py``) and
materialising a *(original_feature, clean_prompt, corrupted_prompt)* row per
matching caption. These rows are the inputs to activation patching in
``src/patch_layers.py``.

Usage::

    # Regenerate the CSV if it's missing, then push.
    python src/preprocess/prepare_prompts.py --limit 256
    python scripts/hub/push_localization_prompts.py \\
        --repo-id lukasz-staniszewski/patching-music-musiccaps-prompts \\
        --csv data/generated_prompts.csv

After upload::

    from datasets import load_dataset
    ds = load_dataset("lukasz-staniszewski/patching-music-musiccaps-prompts")
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _render_dataset_card(repo_id: str, features: list[str], tags, n_rows: int) -> str:
    yaml_tags = "\n".join(f"  - {t}" for t in sorted(set(tags)))
    feature_list = ", ".join(f"`{f}`" for f in features)
    return (
        f"---\n"
        f"language:\n  - en\n"
        f"tags:\n{yaml_tags}\n"
        f"---\n\n"
        f"# Activation-patching prompt pairs (MusicCaps-derived)\n\n"
        f"{n_rows} rows of *(clean, corrupted)* prompt pairs derived from the "
        f"[MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) captions "
        f"by swapping feature-bearing words (e.g. *violin*↔*trumpet*, "
        f"*female*↔*male*, *fast*↔*slow*) using the mapping in "
        f"`src/preprocess/features.py`.\n\n"
        f"Features covered ({len(features)}): {feature_list}.\n\n"
        f"## Schema\n"
        f"- `original_feature`: feature being localised (filter on this for a per-feature run).\n"
        f"- `clean_prompt`: original MusicCaps caption mentioning the feature.\n"
        f"- `corrupted_prompt`: caption with feature-bearing words swapped for the counterfactual.\n\n"
        f"Use the *clean* prompt to collect the un-patched activations and the *corrupted* prompt "
        f"to source the activations that get patched in. Counterfactuals come from "
        f"`MUSICCAPS_SWAPS_FEATURES` in `src/preprocess/features.py`.\n\n"
        f"## Quickstart\n\n"
        f"```python\n"
        f"from datasets import load_dataset\n"
        f"ds = load_dataset({repo_id!r})\n"
        f"violin_rows = ds['train'].filter(lambda r: r['original_feature'] == 'violin')\n"
        f"```\n\n"
        f"## Regenerating from source\n\n"
        f"```bash\n"
        f"python src/preprocess/prepare_prompts.py \\\n"
        f"    --input_file data/music_caps.csv \\\n"
        f"    --limit 256 \\\n"
        f"    --output_file data/generated_prompts.csv\n"
        f"```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="<org>/<dataset-name>")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/generated_prompts.csv"),
        help="Path to the prompts CSV produced by src/preprocess/prepare_prompts.py.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=("audio", "music", "musiccaps", "activation-patching", "localization", "counterfactual"),
    )
    parser.add_argument("--commit-message", default="Upload MusicCaps-derived localization prompts")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"Missing prompts CSV: {args.csv}. "
            f"Run `python src/preprocess/prepare_prompts.py` first."
        )

    import pandas as pd
    from datasets import Dataset

    df = pd.read_csv(args.csv)
    expected = {"original_feature", "clean_prompt", "corrupted_prompt"}
    if not expected.issubset(df.columns):
        raise SystemExit(f"CSV is missing required columns. Got: {list(df.columns)}.")

    features = sorted(df["original_feature"].unique().tolist())
    print(f"Loaded {len(df)} rows covering {len(features)} features: {features}")

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )

    from huggingface_hub import HfApi

    card = _render_dataset_card(args.repo_id, features, args.tags, len(df))
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"Uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
