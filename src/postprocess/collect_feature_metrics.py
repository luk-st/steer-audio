import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd

from src.constants import OUTPUTS_DIR

DEFAULT_FILENAME = "metrics.csv"


def process_alignment_metrics(metrics_data: Dict) -> Dict:
    """Extract alignment metrics and convert them to a dictionary."""
    return {metric: float(value) for metric, value in metrics_data["alignment"].items()}


def process_clap_metrics(metrics_data: Dict) -> Tuple:
    """
    Extract CLAP metrics and prompts, returning them separately for later use.

    Returns:
        metrics (Dict): A dictionary of CLAP metrics with indexed column names.
        prompts (Dict): A dictionary of CLAP prompts with indexed column names.
    """
    clap_metrics = {}
    clap_music_metrics = {}
    clap_eval_prompts = {}

    # clap metrics
    for idx, (prompt, stats) in enumerate(metrics_data["clap"].items(), start=1):
        for stat_type, value in stats.items():
            column_name = f"clap_{idx}_{stat_type}"
            clap_metrics[column_name] = float(value)
        clap_eval_prompts[f"clap_prompt_{idx}"] = prompt

    if "clap_music" in metrics_data.keys():
        for idx, (prompt, stats) in enumerate(metrics_data["clap_music"].items(), start=1):
            for stat_type, value in stats.items():
                column_name = f"clap_music_{idx}_{stat_type}"
                clap_music_metrics[column_name] = float(value)

    return clap_metrics, clap_music_metrics, clap_eval_prompts

def process_muqt_metrics(metrics_data: Dict) -> Tuple:
    """
    Extract MUQ-T metrics and prompts, returning them separately for later use.

    Returns:
        metrics (Dict): A dictionary of MUQ-T metrics with indexed column names.
        prompts (Dict): A dictionary of MUQ-T prompts with indexed column names.
    """
    muqt_metrics = {}
    muqt_eval_prompts = {}

    # muqt metrics
    for idx, (prompt, stats) in enumerate(metrics_data["muqt"].items(), start=1):
        for stat_type, value in stats.items():
            column_name = f"muqt_{idx}_{stat_type}"
            muqt_metrics[column_name] = float(value)
        muqt_eval_prompts[f"muqt_prompt_{idx}"] = prompt

    return muqt_metrics, muqt_eval_prompts


def main(feature: str, blocks_to_path: List[str], output_dir: str, filename: str, model_name: str, localization: str) -> pd.DataFrame:
    """Process metrics for a feature across blocks and save to a CSV file.

    Args:
        feature (str): The feature name to process.
        blocks_to_path (List[str]): A list of blocks to iterate over.
        output_dir (str): The feature output directory where results will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the processed metrics.
    """
    rows = []
    feature_out_path = os.path.join(output_dir, model_name, localization, feature)
    os.makedirs(feature_out_path, exist_ok=True)

    for block in blocks_to_path:
        metric_file_path = os.path.join(feature_out_path, block, "metrics.json")
        with open(metric_file_path, "r") as f:
            metrics_data = json.load(f)

        row = {"Block": block}
        old_row = row.copy()
        try:
            row.update(process_alignment_metrics(metrics_data))
        except KeyError:
            row = old_row
        if "clap" in metrics_data.keys():
            clap_metrics, clap_music_metrics, clap_eval_prompts = process_clap_metrics(metrics_data)
            row.update(clap_metrics)
            if clap_music_metrics != {}:
                row.update(clap_music_metrics)
            row.update(clap_eval_prompts)
        if "muqt" in metrics_data.keys():
            muqt_metrics, muqt_eval_prompts = process_muqt_metrics(metrics_data)
            row.update(muqt_metrics)
            row.update(muqt_eval_prompts)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv_path = os.path.join(feature_out_path, filename)
    df.to_csv(output_csv_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all blocks results.")
    parser.add_argument("--feature", type=str, help="Input file to load musiccaps.")
    parser.add_argument("--blocks_to_path", type=str, nargs="+", help="Number of prompts to generate per feature.")
    parser.add_argument("--output_dir", type=str, default=OUTPUTS_DIR, help="Output file dir to save results.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Output file name to save results.")
    parser.add_argument("--model_name", type=str, help="Name of the model.")
    parser.add_argument("--localization", type=str, choices=["patching", "ablate"], help="Type of the localization method.", default="patching")
    args = parser.parse_args()

    main(args.feature, args.blocks_to_path, args.output_dir, args.filename, args.model_name, args.localization)
