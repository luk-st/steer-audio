#!/usr/bin/env python3
"""
Compute AUC-based steering evaluation metrics from protocol_results directories.

Alignment metrics (MuQ-T, CLAP):
  - X axis: LPAPS (integrated in sweep order, i.e. increasing |alpha|)
  - Y axis: delta — sign-corrected so higher = better:
      pos: delta = value(alpha) - baseline
      neg: delta = baseline - value(alpha)
  - AUC = trapezoid integral on raw alpha points
  - Wrong-direction steering contributes negatively (penalizes overshooting)
  - Reported as: pos / neg / avg=(pos+neg)/2

Smoothness metrics (CSM, extended from FreeSliders):
  - Alphas truncated to those with LPAPS <= global_cutoff (fair comparison)
  - Sign-correct alignment so "increasing" always means better
  - Normalize to [0,1] by dividing by max(delta) across the direction
  - Compute consecutive gaps (including negative = non-monotonic penalty)
  - CSM = std(gaps), lower = better
  - Reported as: pos / neg / avg=(pos+neg)/2

Quality metrics (CE, CU, PC, PQ):
  - Sample quality at N_QUALITY_POINTS uniformly spaced LPAPS values on a common grid
  - Average all 2*N_QUALITY_POINTS values (pos + neg) → one scalar per metric
  - Raw mean (not delta)

Output columns (--direction both):
  muqt_pos, muqt_neg, muqt_avg,
  clap_pos, clap_neg, clap_avg,
  csm_muqt_pos, csm_muqt_neg, csm_muqt_avg,
  csm_clap_pos, csm_clap_neg, csm_clap_avg,
  CE, CU, PC, PQ

Usage:
    python auc.py PATH1 PATH2 ... --direction both --labels "PI" "SAE" ...
"""

import argparse
import glob
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from numpy import trapezoid as _trapezoid
except ImportError:
    from numpy import trapz as _trapezoid  # type: ignore

QUALITY_COLS = [
    "content_enjoyment_mean",
    "content_usefulness_mean",
    "production_complexity_mean",
    "production_quality_mean",
]
QUALITY_SHORT = ["CE", "CU", "PC", "PQ"]
DEFAULT_ALIGNMENT_METRICS = ["muqt", "clap"]
N_QUALITY_POINTS = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def filter_direction(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "pos":
        return df[df["alpha"] >= 0].copy()
    return df[df["alpha"] <= 0].copy()


def get_baseline(df: pd.DataFrame, col: str) -> float:
    row = df[np.isclose(df["alpha"], 0.0)]
    if row.empty:
        row = df.iloc[[df["alpha"].abs().idxmin()]]
    return float(row[col].iloc[0])


def compute_auc(
    lpaps: np.ndarray,
    delta: np.ndarray,
    cutoff: float,
    abs_alpha: Optional[np.ndarray] = None,
) -> float:
    """Integrate delta over LPAPS up to cutoff, following the sweep.

    Points are traversed in order of increasing ``|alpha|`` — the order the sweep
    actually produced them — so this is the path integral of ``delta`` over LPAPS.
    Where LPAPS backtracks (it is not perfectly monotone in ``|alpha|``), the
    backward segment subtracts area and the following forward segment re-adds it,
    which is the faithful treatment. Sorting by LPAPS instead would silently
    reorder the sweep into a monotone curve that was never measured.
    """
    mask = lpaps <= cutoff + 1e-9
    lp, dv = lpaps[mask], delta[mask]
    if len(lp) < 2:
        return float("nan")
    if abs_alpha is not None:
        order = np.argsort(abs_alpha[mask], kind="stable")
        lp, dv = lp[order], dv[order]
    return float(_trapezoid(dv, lp))


def get_lpaps_max(results_dir: Path, direction: str) -> float:
    df = load_csv_safe(results_dir / "lpaps.csv")
    if df is None:
        return float("nan")
    return float(filter_direction(df, direction)["mean"].max())


def global_cutoff(dirs: List[Path], direction: str) -> float:
    valid = [m for m in (get_lpaps_max(d, direction) for d in dirs)
             if not math.isnan(m)]
    return float(min(valid)) if valid else float("nan")


def pci_cutoff(root: Path, concept: str, direction: str) -> float:
    """Paper cutoff: min(max-LPAPS of PCI-all, PCI-loc) for one direction.

    PCI defines the shared preservation ceiling; whichever of the global
    (``pci_all``) or localized (``pci_loc``) PCI run reaches the lower max
    LPAPS at its strongest alpha sets the cutoff for that direction.
    """
    root = Path(root)
    dirs = [root / f"pci_all_{concept}" / "protocol_results",
            root / f"pci_loc_{concept}" / "protocol_results"]
    dirs = [d for d in dirs if (d / "lpaps.csv").exists()]
    return global_cutoff(dirs, direction)


def fmt(v, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "?.??"
    return f"{v:.{decimals}f}"


def load_and_merge_alignment(
    results_dir: Path,
    direction: str,
    metric: str,
    lpaps_cutoff: float,
    add_cutoff_boundary: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Load lpaps and alignment CSVs, merge on alpha, filter to direction,
    and truncate to alphas where LPAPS <= lpaps_cutoff.

    If ``add_cutoff_boundary``, append one interpolated row at exactly
    ``lpaps == lpaps_cutoff`` — ``val`` linearly interpolated between the last
    alpha below the cutoff and the first alpha above it — so the AUC integrates
    over exactly [0, cutoff] and the final [14th-alpha -> cutoff] segment is
    included (identical preservation interval across methods). Requires a point
    above the cutoff to bracket it; if none exists the curve is left as-is.
    Returns DataFrame with columns: alpha, lpaps, val or None if missing/empty.
    """
    lpaps_df = load_csv_safe(results_dir / "lpaps.csv")
    align_df = load_csv_safe(results_dir / f"{metric}.csv")
    if lpaps_df is None or align_df is None:
        return None

    lpaps_df = filter_direction(lpaps_df, direction).sort_values("alpha")
    align_df = filter_direction(align_df, direction).sort_values("alpha")
    lpaps_df["alpha"] = lpaps_df["alpha"].round(6)
    align_df["alpha"] = align_df["alpha"].round(6)

    merged = lpaps_df[["alpha", "mean"]].rename(columns={"mean": "lpaps"}).merge(
        align_df[["alpha", "mean"]].rename(columns={"mean": "val"}), on="alpha"
    )
    if merged.empty:
        return None

    below = merged[merged["lpaps"] <= lpaps_cutoff + 1e-9]
    if below.empty:
        return None
    above = merged[merged["lpaps"] > lpaps_cutoff + 1e-9]

    if add_cutoff_boundary and not above.empty:
        b = below.sort_values("lpaps").iloc[-1]   # last alpha below cutoff
        a = above.sort_values("lpaps").iloc[0]    # first alpha above cutoff
        if a["lpaps"] - b["lpaps"] > 1e-12:
            frac = (lpaps_cutoff - b["lpaps"]) / (a["lpaps"] - b["lpaps"])
            val_cut = float(b["val"]) + frac * (float(a["val"]) - float(b["val"]))
            boundary = pd.DataFrame([{"alpha": a["alpha"], "lpaps": lpaps_cutoff, "val": val_cut}])
            below = pd.concat([below, boundary], ignore_index=True)

    return below.sort_values("alpha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Alignment AUC — one direction
# ---------------------------------------------------------------------------

def compute_alignment_auc_direction(
    results_dir: Path,
    direction: str,
    cutoff: float,
    alignment_metrics: List[str],
) -> Dict[str, float]:
    """AUC for alignment metrics, delta sign-corrected, clipped at 0."""
    out: Dict[str, float] = {}
    sign = 1.0 if direction == "pos" else -1.0

    lpaps_df = load_csv_safe(results_dir / "lpaps.csv")
    if lpaps_df is None:
        warnings.warn(f"lpaps.csv not found in {results_dir}")
        return out

    lpaps_df = filter_direction(lpaps_df, direction).sort_values("alpha")
    out["lpaps_max"] = float(lpaps_df["mean"].max())

    for metric in alignment_metrics:
        merged = load_and_merge_alignment(
            results_dir, direction, metric, cutoff, add_cutoff_boundary=True
        )
        if merged is None:
            continue
        baseline = float(merged.loc[merged["alpha"].abs().idxmin(), "val"])
        delta = sign * (merged["val"].values - baseline)
        out[metric] = compute_auc(
            merged["lpaps"].values, delta, cutoff,
            abs_alpha=merged["alpha"].abs().values,
        )

    return out


# ---------------------------------------------------------------------------
# Smoothness (CSM) — one direction
# ---------------------------------------------------------------------------

def compute_csm_direction(
    results_dir: Path,
    direction: str,
    cutoff: float,
    alignment_metrics: List[str],
) -> Dict[str, float]:
    """
    CSM = std of consecutive gaps in normalized delta alignment.

    Steps:
      1. Load alignment, truncate to alphas with LPAPS <= cutoff
      2. Sign-correct delta (same as AUC): pos: val-baseline, neg: baseline-val
      3. Normalize to [0,1] by dividing by max(delta); if max<=0, return nan
      4. Sort by alpha (ascending absolute value, i.e. 0 -> alpha_max)
      5. Compute consecutive gaps (including negative gaps)
      6. CSM = std(gaps), lower = better
    """
    out: Dict[str, float] = {}
    sign = 1.0 if direction == "pos" else -1.0

    for metric in alignment_metrics:
        merged = load_and_merge_alignment(results_dir, direction, metric, cutoff)
        if merged is None:
            continue

        baseline = float(merged.loc[merged["alpha"].abs().idxmin(), "val"])
        delta = sign * (merged["val"].values - baseline)

        # Normalize by max clipped delta (correct-direction peak only)
        # In classic cases max(clipped) == max(delta), so behaviour is identical
        delta_clipped = np.clip(delta, 0, None)
        a_max = float(np.max(delta_clipped))
        if a_max <= 1e-9:
            out[f"csm_{metric}"] = float("nan")
            continue

        # Normalize full delta (including negatives) by correct-direction peak
        delta_norm = delta / a_max

        # Sort by increasing |alpha| (from 0 to alpha_max)
        abs_alpha = merged["alpha"].abs().values
        order = np.argsort(abs_alpha)
        delta_sorted = delta_norm[order]

        # Consecutive gaps (including negative = non-monotonic penalty)
        gaps = np.diff(delta_sorted)
        if len(gaps) == 0:
            out[f"csm_{metric}"] = float("nan")
            continue

        out[f"csm_{metric}"] = float(np.std(gaps))

    return out


# ---------------------------------------------------------------------------
# Quality metric — interpolated mean over common LPAPS grid
# ---------------------------------------------------------------------------

def interpolate_quality_at_lpaps(
    results_dir: Path,
    direction: str,
    lpaps_grid: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    lpaps_df = load_csv_safe(results_dir / "lpaps.csv")
    aes_df = load_csv_safe(results_dir / "aesthetics.csv")
    if lpaps_df is None or aes_df is None:
        return out

    lpaps_df = filter_direction(lpaps_df, direction).sort_values("alpha")
    aes_df = filter_direction(aes_df, direction).sort_values("alpha")
    lpaps_df["alpha"] = lpaps_df["alpha"].round(6)
    aes_df["alpha"] = aes_df["alpha"].round(6)

    merged = lpaps_df[["alpha", "mean"]].rename(columns={"mean": "lpaps"}).merge(
        aes_df[["alpha"] + QUALITY_COLS], on="alpha"
    )
    if merged.empty:
        return out

    lpaps_vals = merged["lpaps"].values

    for col, short in zip(QUALITY_COLS, QUALITY_SHORT):
        if col not in merged.columns:
            continue
        quality_vals = merged[col].values
        order = np.argsort(lpaps_vals)
        lp_sorted = lpaps_vals[order]
        q_sorted = quality_vals[order]
        grid_in_range = lpaps_grid[
            (lpaps_grid >= lp_sorted[0] - 1e-9) &
            (lpaps_grid <= lp_sorted[-1] + 1e-9)
        ]
        if len(grid_in_range) == 0:
            continue
        interpolated = np.interp(grid_in_range, lp_sorted, q_sorted)
        out[short] = float(np.mean(interpolated))

    return out


def compute_quality_metrics(
    results_dirs: List[Path],
    labels: List[str],
    pos_cutoff: float,
    neg_cutoff: float,
    n_points: int,
    latex_only: bool,
) -> Dict[str, Dict[str, float]]:
    pos_grid = np.linspace(0, pos_cutoff, n_points)
    neg_grid = np.linspace(0, neg_cutoff, n_points)

    if not latex_only:
        print(f"  Quality LPAPS grid (pos): {n_points} points in [0, {pos_cutoff:.4f}]")
        print(f"  Quality LPAPS grid (neg): {n_points} points in [0, {neg_cutoff:.4f}]")

    results: Dict[str, Dict[str, float]] = {}
    for rdir, label in zip(results_dirs, labels):
        pos_q = interpolate_quality_at_lpaps(rdir, "pos", pos_grid)
        neg_q = interpolate_quality_at_lpaps(rdir, "neg", neg_grid)

        combined: Dict[str, float] = {}
        for short in QUALITY_SHORT:
            pv = pos_q.get(short)
            nv = neg_q.get(short)
            values = [v for v in [pv, nv] if v is not None and not math.isnan(v)]
            combined[short] = float(np.mean(values)) if values else float("nan")

        results[label] = combined

    return results


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def build_rows(
    results_dirs: List[Path],
    labels: List[str],
    directions: List[str],
    alignment_metrics: List[str],
    latex_only: bool,
) -> List[str]:
    do_avg = set(directions) == {"pos", "neg"}

    align_cutoffs = {d: global_cutoff(results_dirs, d) for d in directions}
    if not latex_only:
        for d, c in align_cutoffs.items():
            print(f"  Global LPAPS cutoff ({d}): {c:.4f}")

    # Alignment AUC
    align_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rdir, label in zip(results_dirs, labels):
        align_results[label] = {}
        for d in directions:
            align_results[label][d] = compute_alignment_auc_direction(
                rdir, d, align_cutoffs[d], alignment_metrics
            )
            if not latex_only:
                lmax = align_results[label][d].get("lpaps_max", float("nan"))
                print(f"  {label} ({d}): LPAPS_max={lmax:.3f}")

    # Smoothness CSM
    csm_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rdir, label in zip(results_dirs, labels):
        csm_results[label] = {}
        for d in directions:
            csm_results[label][d] = compute_csm_direction(
                rdir, d, align_cutoffs[d], alignment_metrics
            )

    # Quality
    pos_cutoff = align_cutoffs.get("pos", global_cutoff(results_dirs, "pos"))
    neg_cutoff = align_cutoffs.get("neg", global_cutoff(results_dirs, "neg"))
    quality_results = compute_quality_metrics(
        results_dirs, labels, pos_cutoff, neg_cutoff, N_QUALITY_POINTS, latex_only
    )

    rows = []
    n_rows = len(labels)
    for i, label in enumerate(labels):
        vals = []

        # Alignment AUC: pos / neg / avg
        for metric in alignment_metrics:
            if do_avg:
                pos_v = align_results[label]["pos"].get(metric)
                neg_v = align_results[label]["neg"].get(metric)
                vals.append(fmt(pos_v))
                vals.append(fmt(neg_v))
                avg = (pos_v + neg_v) / 2.0 if (
                    pos_v is not None and neg_v is not None
                    and not math.isnan(pos_v) and not math.isnan(neg_v)
                ) else float("nan")
                vals.append(fmt(avg))
            else:
                vals.append(fmt(align_results[label][directions[0]].get(metric)))

        # Smoothness CSM: pos / neg / avg
        for metric in alignment_metrics:
            key = f"csm_{metric}"
            if do_avg:
                pos_v = csm_results[label]["pos"].get(key)
                neg_v = csm_results[label]["neg"].get(key)
                vals.append(fmt(pos_v))
                vals.append(fmt(neg_v))
                avg = (pos_v + neg_v) / 2.0 if (
                    pos_v is not None and neg_v is not None
                    and not math.isnan(pos_v) and not math.isnan(neg_v)
                ) else float("nan")
                vals.append(fmt(avg))
            else:
                vals.append(fmt(csm_results[label][directions[0]].get(key)))

        # Quality — single averaged scalar across CE/CU/PC/PQ
        q_vals = [quality_results[label].get(s) for s in QUALITY_SHORT]
        q_vals = [v for v in q_vals if v is not None and not math.isnan(v)]
        vals.append(fmt(float(np.mean(q_vals)) if q_vals else float("nan")))

        latex_vals = " & ".join(f"${v}$" for v in vals)
        latex_label = label.replace("{7,8}", r"$\mathbf{\{7,8\}}$")
        rows.append(f"& {latex_label} & {latex_vals}\\\\")

    return rows


def print_header(alignment_metrics: List[str], directions: List[str]) -> None:
    do_avg = set(directions) == {"pos", "neg"}
    parts = []
    for metric in alignment_metrics:
        if do_avg:
            parts.append(f"AUC_{metric}: +/-/avg")
        else:
            parts.append(f"AUC_{metric}_{directions[0]}")
    for metric in alignment_metrics:
        if do_avg:
            parts.append(f"CSM_{metric}: +/-/avg")
        else:
            parts.append(f"CSM_{metric}_{directions[0]}")
    parts.append("Quality_avg")
    print("Columns: " + " | ".join(parts))
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_label_from_path(path: Path) -> str:
    parts = path.parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "outputs")
        return parts[idx + 1] if len(parts) > idx + 1 else "?"
    except (StopIteration, IndexError):
        return path.parent.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AUC + CSM steering evaluation"
    )
    parser.add_argument("paths", nargs="+", help="protocol_results dirs (globs ok)")
    parser.add_argument(
        "--direction", choices=["pos", "neg", "both"], default="both",
    )
    parser.add_argument(
        "--alignment_metrics", nargs="+", default=DEFAULT_ALIGNMENT_METRICS,
    )
    parser.add_argument(
        "--n_quality_points", type=int, default=N_QUALITY_POINTS,
    )
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--auto_label", action="store_true")
    parser.add_argument("--latex_only", action="store_true")
    args = parser.parse_args()

    all_paths: List[Path] = []
    for pattern in args.paths:
        expanded = sorted(glob.glob(pattern, recursive=True))
        if expanded:
            if len(expanded) > 1 and not args.latex_only:
                print(f"Warning: '{pattern}' matched {len(expanded)} paths, using first")
            all_paths.append(Path(expanded[0]))
        else:
            p = Path(pattern.rstrip("/"))
            if p.exists():
                all_paths.append(p)
            elif not args.latex_only:
                print(f"Warning: no match for '{pattern}'")

    if not all_paths:
        print("Error: no valid paths found")
        return 1

    if args.labels:
        if len(args.labels) != len(all_paths):
            print(f"Error: --labels count ({len(args.labels)}) != paths ({len(all_paths)})")
            return 1
        labels = args.labels
    elif args.auto_label or len(all_paths) > 1:
        labels = [get_label_from_path(p) for p in all_paths]
    else:
        labels = [str(all_paths[0])]

    directions = ["pos", "neg"] if args.direction == "both" else [args.direction]

    if not args.latex_only:
        print_header(args.alignment_metrics, directions)

    rows = build_rows(
        all_paths, labels, directions,
        args.alignment_metrics, args.latex_only
    )

    if not args.latex_only:
        print("\n--- LaTeX rows ---")
    for row in rows:
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())