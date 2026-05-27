import argparse
import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from src.preprocess.features import (
    MUSICCAPS_COUNTERFACTUAL_FEATURES,
    MUSICCAPS_ORIGINAL_FEATURES,
    MUSICCAPS_SWAPS_FEATURES,
)

MUSIC_CAPS = "data/music_caps.csv"
DEFAULT_LIMIT = 256
DEFAULT_OUTPUT_FILE = "data/generated_prompts.csv"

FEATURES_TO_PROCESS = [
    "female",
    "male",
    "fast",
    "slow",
    "happy",
    "sad",
    "reggae",
    "metal",
    "opera",
    "jazz",
    "violin",
    "trumpet",
    "saxophone",
    "drums",
    "cello",
    "bongos",
    "flute",
    "maracas",
    "harmonica",
    "trombone",
    "xylophone",
]


def _swap_prompt(prompt: str, swap_map: Dict[str, str]) -> str:
    corrupted_prompt = prompt
    for old_word, new_word in swap_map.items():
        corrupted_prompt = re.sub(
            rf"\b{old_word}\b", new_word, corrupted_prompt, flags=re.IGNORECASE
        )
    return corrupted_prompt


def generate_prompts(
    feature_words: Union[str, List[str]],
    text_column: str,
    prompts_df: pd.DataFrame,
    counterfactual_words: Union[str, List[str], None],
    swap_map: Union[Dict[str, str], None] = None,
    limit=-1,
) -> List[Dict]:
    results = []
    count = 0

    if isinstance(feature_words, str):
        feature_words = [feature_words]
    if isinstance(counterfactual_words, str):
        counterfactual_words = [counterfactual_words]
    elif counterfactual_words is None:
        counterfactual_words = []

    for _, row in prompts_df.iterrows():
        if limit != -1 and count >= limit:
            break
        caption = row[text_column]
        if any(
            re.search(rf"\b{word}\b", caption, flags=re.IGNORECASE)
            for word in feature_words
        ) and not any(
            re.search(rf"\b{word}\b", caption, flags=re.IGNORECASE)
            for word in counterfactual_words
        ):
            clean_caption = re.sub(r"\n", " ", caption)

            corrupted_caption = ""
            if swap_map is not None:
                corrupted_caption = _swap_prompt(
                    prompt=clean_caption, swap_map=swap_map
                )

            results.append(
                {
                    "original_feature": feature_words[0],
                    "clean_prompt": clean_caption,
                    "corrupted_prompt": corrupted_caption,
                }
            )
            count += 1
    return results


def main(limit: int, output_file: str, input_file: str = MUSIC_CAPS) -> None:
    data = pd.read_csv(input_file)
    results = []

    for feature in FEATURES_TO_PROCESS:
        features = MUSICCAPS_ORIGINAL_FEATURES[feature]
        counterfactual_features = MUSICCAPS_COUNTERFACTUAL_FEATURES[feature]
        swap_map = MUSICCAPS_SWAPS_FEATURES[feature]
        feature_results = generate_prompts(
            feature_words=features,
            text_column="caption",
            prompts_df=data,
            counterfactual_words=counterfactual_features,
            swap_map=swap_map,
            limit=limit,
        )
        results.extend(feature_results)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save prompts with word replacements."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=MUSIC_CAPS,
        help="Input file to load musiccaps.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of prompts to generate per feature.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output file name to save results.",
    )
    args = parser.parse_args()

    main(args.limit, args.output_file, args.input_file)
