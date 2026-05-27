import argparse
from pathlib import Path
import pandas as pd

METRICS_AESTHETICS = ["CE", "CU", "PC", "PQ"]


def main(file_in: str, file_out: str, tf_name: str):
    assert file_in.endswith(".jsonl")
    file_in = Path(file_in)
    file_out = Path(file_out)

    df = pd.read_json(file_in, lines=True)
    results = {"name": tf_name}
    for metric in METRICS_AESTHETICS:
        results[f"{metric}_mean"] = df[metric].mean()
        results[f"{metric}_std"] = df[metric].std()
    if file_out.exists():
        df_results = pd.read_csv(file_out)
        df_results = pd.concat([df_results, pd.DataFrame(results, index=[0])], ignore_index=True)
    else:
        df_results = pd.DataFrame(results, index=[0])
    df_results.to_csv(file_out, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_in", type=str, required=True)
    parser.add_argument("--file_out", type=str, required=True)
    parser.add_argument("--tf_name", type=str, required=True)
    args = parser.parse_args()
    main(args.file_in, args.file_out, args.tf_name)