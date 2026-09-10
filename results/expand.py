"""Turn `steering_metrics.csv` + `quality_metrics.csv` into an eval root.

The published numbers ship as a few long-format CSVs, but the evaluation code
reads a directory per method x concept. This rebuilds that layout -- by default
into a temporary directory that the consumer scripts create on demand, so
nothing large is kept on disk.

    python results/expand.py --dest /tmp/eval_root    # materialise explicitly
"""

import tempfile
from pathlib import Path

import pandas as pd

AES = ["content_enjoyment", "content_usefulness",
       "production_complexity", "production_quality"]


def expand(results_dir: str = "results", dest: str = None) -> Path:
    """Write the protocol_results tree and return its root.

    Args:
        results_dir: directory holding the consolidated CSVs.
        dest: where to write; a temporary directory when omitted.
    """
    src = Path(results_dir)
    root = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="steer_results_"))
    metrics = pd.read_csv(src / "steering_metrics.csv")
    quality = pd.read_csv(src / "quality_metrics.csv")

    for (cell, concept, axis), g in metrics.groupby(["cell", "concept", "axis"]):
        sub = "protocol_results" if axis == "lpaps" else f"protocol_results_{axis}"
        d = root / f"{cell}_{concept}" / sub
        d.mkdir(parents=True, exist_ok=True)
        g = g.sort_values("alpha")
        g[["alpha", "lpaps"]].rename(columns={"lpaps": "mean"}).to_csv(
            d / "lpaps.csv", index=False)
        for m in ("muqt", "clap"):
            if g[m].notna().any():
                g[["alpha", m]].rename(columns={m: "mean"}).to_csv(
                    d / f"{m}.csv", index=False)

    for (cell, concept), g in quality.groupby(["cell", "concept"]):
        d = root / f"{cell}_{concept}" / "protocol_results"
        d.mkdir(parents=True, exist_ok=True)
        g = g.sort_values("alpha")[["alpha"] + AES]
        g.columns = ["alpha"] + [f"{a}_mean" for a in AES]
        g.to_csv(d / "aesthetics.csv", index=False)

    return root


if __name__ == "__main__":
    import fire

    fire.Fire(lambda results_dir="results", dest=None: print(expand(results_dir, dest)))
