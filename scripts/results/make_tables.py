"""Reproduce the paper's benchmark tables from the shipped metrics.

Reads `results/eval` (per-alpha LPAPS / MuQ / CLAP / aesthetics for every
method x concept) and applies the published protocol via
`src/steering/eval/auc.py`: per concept and direction the preservation cutoff is
min(PCI-all, PCI-loc) max-LPAPS, curves are truncated there, and AUC is the
trapezoid area. Writes markdown.

    uv run python scripts/results/make_tables.py                  # per-concept + avg
    uv run python scripts/results/make_tables.py --axis harmony   # a MIR axis
    uv run python scripts/results/make_tables.py --localization   # layer impact
"""

import importlib.util
from pathlib import Path

import fire
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
# (label, cell prefix); localized arms are marked (loc.)
METHODS = [("PCI", "pci_all"), ("PCI (loc.)", "pci_loc"),
           ("Text Embeddings", "textemb_all"), ("Text Embeddings (loc.)", "textemb_loc"),
           ("Token Embeddings", "tokemb_all"), ("Token Embeddings (loc.)", "tokemb_loc"),
           ("FreeSliders", "freesliders_all"), ("FreeSliders (loc.)", "freesliders_loc"),
           ("Concept Sliders", "cs_all"), ("Concept Sliders (loc.)", "cs_loc"),
           ("AUSteer", "austeer_all"), ("AUSteer (loc.)", "austeer_loc"),
           ("CAA", "caa_all"), ("CAA (loc.)", "caa_loc"),
           ("SAE (loc.)", "sae")]
METRICS = ["muqt", "clap"]


def _sub(axis: str) -> str:
    return "protocol_results" if axis is None else f"protocol_results_{axis}"


def _rows(root: Path, concept: str, axis: str):
    sub = _sub(axis)
    present = [(lbl, root / f"{p}_{concept}") for lbl, p in METHODS
               if (root / f"{p}_{concept}" / sub / "lpaps.csv").exists()]
    cuts = {}
    for d in ("pos", "neg"):
        dirs = [root / f"pci_{v}_{concept}" / sub for v in ("all", "loc")]
        cuts[d] = auc.global_cutoff([x for x in dirs if (x / "lpaps.csv").exists()], d)
    qual = auc.compute_quality_metrics([d / sub for _, d in present],
                                       [l for l, _ in present],
                                       cuts["pos"], cuts["neg"],
                                       auc.N_QUALITY_POINTS, True)
    out = []
    for lbl, d in present:
        r = {"method": lbl}
        pos = auc.compute_alignment_auc_direction(d / sub, "pos", cuts["pos"], METRICS)
        neg = auc.compute_alignment_auc_direction(d / sub, "neg", cuts["neg"], METRICS)
        csp = auc.compute_csm_direction(d / sub, "pos", cuts["pos"], METRICS)
        csn = auc.compute_csm_direction(d / sub, "neg", cuts["neg"], METRICS)
        for m in METRICS:
            r[f"auc_{m}"] = _avg(pos.get(m), neg.get(m))
            r[f"csm_{m}"] = _avg(csp.get(f"csm_{m}"), csn.get(f"csm_{m}"))
        q = [v for v in qual[lbl].values() if v is not None and v == v]
        r["quality"] = sum(q) / len(q) if q else None
        out.append(r)
    return pd.DataFrame(out), cuts


def _avg(a, b):
    return (a + b) / 2 if (a is not None and b is not None) else None


def _md(df: pd.DataFrame) -> str:
    cols = ["method", "auc_muqt", "auc_clap", "csm_muqt", "csm_clap", "quality"]
    head = "| Method | AUC MuQ | AUC CLAP | Smooth MuQ | Smooth CLAP | Quality |"
    lines = [head, "|" + "---|" * 6]
    for _, r in df.iterrows():
        f = lambda v: "-" if v is None or v != v else f"{v:.3f}"
        lines.append("| " + " | ".join([r["method"]] + [f(r[c]) for c in cols[1:]]) + " |")
    return "\n".join(lines)


def main(root: str = "results", axis: str = None, localization: bool = False,
         out: str = None):
    """Print the tables (and write them if `out` is given).

    Args:
        root: the results dir (expanded automatically) or an eval root.
        axis: MIR axis (harmony/melody/rhythm/ssm); default is LPAPS preservation.
        localization: print the layer-impact table instead.
        out: optional markdown file to write.
    """
    if localization:
        imp = pd.read_csv(Path(root) / "localization_impact.csv")
        agg = (imp.groupby(["model", "block", "attn"])["impact"].mean()
                  .reset_index().sort_values(["model", "impact"], ascending=[True, False]))
        text = ["# Layer impact I(l), averaged over concepts", ""]
        for model, g in agg.groupby("model"):
            text += [f"## {model}", "", "| block | attn | I(l) |", "|---|---|---|"]
            text += [f"| {r.block} | {r.attn} | {r.impact:.3f} |" for r in g.itertuples()]
            text += [""]
        body = "\n".join(text)
    else:
        root = str(_resolve_root(root))
        chunks, avg = [], []
        label = axis or "LPAPS"
        for c in CONCEPTS:
            df, cuts = _rows(Path(root), c, axis)
            if df.empty:
                continue
            chunks += [f"## {c}  (cutoff pos={cuts['pos']:.3f}, neg={cuts['neg']:.3f})",
                       "", _md(df), ""]
            avg.append(df.set_index("method"))
        # mean over the concepts where a cell exists; summing frames instead
        # would propagate a single missing concept into a blank row
        mean = pd.concat(avg).groupby(level=0).mean().reindex(
            [m for m, _ in METHODS])
        body = "\n".join([f"# Benchmark tables — preservation: {label}", "",
                          "## Averaged over the nine concepts", "",
                          _md(mean.reset_index()), ""] + chunks)
    print(body)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(body + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
