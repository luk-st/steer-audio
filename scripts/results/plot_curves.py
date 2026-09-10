"""Plot the alignment-preservation curves from the shipped metrics.

x is preservation (cutoff - LPAPS) and y the sign-corrected alignment delta, so
the shaded area under each curve is exactly the AUC in the tables. The cutoff is
the published one: min(PCI-all, PCI-loc) max-LPAPS for that concept and
direction.

    uv run python scripts/results/plot_curves.py
    uv run python scripts/results/plot_curves.py --concepts piano --axis harmony
"""

import importlib.util
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("auc", _HERE / "src/steering/eval/auc.py")
auc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auc)


def _resolve_root(root: str) -> Path:
    """Accept an eval root, or the consolidated results dir (expanded on demand)."""
    p = Path(root)
    if (p / "steering_metrics.csv").exists():
        spec = importlib.util.spec_from_file_location("expand", p / "expand.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.expand(str(p))
    return p

CONCEPTS = ["piano", "mood", "tempo", "vocal_gender", "vocal_style",
            "violin", "guitar_electronic", "rock_genre", "electronic_music"]
METRIC = {"vocal_gender": "clap"}          # every other concept is scored with MuQ
METHODS = [("PCI", "pci_all"), ("PCI (loc.)", "pci_loc"),
           ("Text Embeddings", "textemb_all"), ("Text Embeddings (loc.)", "textemb_loc"),
           ("Token Embeddings", "tokemb_all"), ("Token Embeddings (loc.)", "tokemb_loc"),
           ("FreeSliders", "freesliders_all"), ("FreeSliders (loc.)", "freesliders_loc"),
           ("Concept Sliders", "cs_all"), ("Concept Sliders (loc.)", "cs_loc"),
           ("AUSteer", "austeer_all"), ("AUSteer (loc.)", "austeer_loc"),
           ("CAA", "caa_all"), ("CAA (loc.)", "caa_loc"),
           ("SAE (loc.)", "sae")]
COLOR = {"PCI": "#f4aa41", "Text Embeddings": "#b8860b", "Token Embeddings": "#8c5a3c",
         "FreeSliders": "#51a851", "Concept Sliders": "#ea5a47", "AUSteer": "#6ea9c7",
         "CAA": "#38789d", "SAE": "#8338eb"}


def _curve(d: Path, metric: str, direction: str, cutoff: float):
    lp = pd.read_csv(d / "lpaps.csv")[["alpha", "mean"]].rename(columns={"mean": "lpaps"})
    al = pd.read_csv(d / f"{metric}.csv")[["alpha", "mean"]].rename(columns={"mean": "val"})
    df = lp.merge(al, on="alpha").sort_values("alpha")
    df = df[df.alpha >= 0] if direction == "pos" else df[df.alpha <= 0]
    if df.empty:
        return None
    base = float(df.loc[df.alpha.abs().idxmin(), "val"])
    df = df[df.lpaps <= cutoff + 1e-9]
    if len(df) < 2:
        return None
    x = cutoff - df.lpaps.to_numpy()
    y = (1 if direction == "pos" else -1) * (df.val.to_numpy() - base)
    order = np.argsort(x)
    return x[order], y[order]


def main(root: str = "results", out_dir: str = "results/curves",
         concepts: str = None, axis: str = None):
    """Write one PDF per concept and direction.

    Args:
        root: the results dir (expanded automatically) or an eval root.
        out_dir: where the figures go.
        concepts: comma-separated subset (default: all nine).
        axis: MIR axis instead of LPAPS preservation.
    """
    root = str(_resolve_root(root))
    sub = "protocol_results" if axis is None else f"protocol_results_{axis}"
    # Fire hands a comma-separated value over as a tuple already
    if concepts is None:
        todo = CONCEPTS
    elif isinstance(concepts, str):
        todo = concepts.split(",")
    else:
        todo = list(concepts)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for concept in todo:
        metric = METRIC.get(concept, "muqt")
        for direction in ("pos", "neg"):
            pci = [Path(root) / f"pci_{v}_{concept}" / sub for v in ("all", "loc")]
            cutoff = auc.global_cutoff([p for p in pci if (p / "lpaps.csv").exists()],
                                       direction)
            fig, ax = plt.subplots(figsize=(8, 6))
            n = 0
            for label, cell in METHODS:
                d = Path(root) / f"{cell}_{concept}" / sub
                if not (d / "lpaps.csv").exists():
                    continue
                c = _curve(d, metric, direction, cutoff)
                if c is None:
                    continue
                base = label.replace(" (loc.)", "")
                ax.plot(*c, linestyle="--" if "(loc.)" in label else "-",
                        color=COLOR[base], marker="o", markersize=4, linewidth=2.0,
                        label=label)
                ax.fill_between(c[0], 0, c[1], color=COLOR[base], alpha=0.08, linewidth=0)
                n += 1
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_xlabel(r"Audio Preservation $\uparrow$", fontsize=13)
            ax.set_ylabel(rf"$\Delta$ Alignment ({metric.upper()}) $\uparrow$", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            path = Path(out_dir) / f"curves_{concept}_{direction}.pdf"
            fig.savefig(path)
            plt.close(fig)
            print(f"{path}  ({n} methods, cutoff={cutoff:.3f})")


if __name__ == "__main__":
    fire.Fire(main)
