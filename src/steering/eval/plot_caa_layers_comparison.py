"""
CAA layer comparison: alignment vs LPAPS curves for all/localized/no-localized.

Two data sources, same plot style:
  - Ace Step CAA  (concept->layer keys: all / tf6tf7 / no_tf6tf7)
  - AudioLDM2 CAA (concept->layer keys: all / loc / ablated)

For each source produces two figures (positive alphas only):
  1. Raw alignment vs LPAPS                       (1x4: piano, mood, tempo, vocal)
  2. Preservation-alignment curve  (delta-align vs preserved-LPAPS)

Concept->metric map: piano/mood/tempo use MUQ-T, vocal_gender uses CLAP.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local-only roots. ACE-Step reads from the cleaned tree under pwr-mount.
ACESTEP_ROOT = "pwr-mount/steering/outputs/ace_step"
# ACESTEP_ROOT = "steering/outputs/ace_step"
AUDIOLDM_ROOT = "steering/outputs_audioldm/no_renorm_steps100_gs45_pos100p"

# Order of subplots, left to right
CONCEPTS = [
    {"key": "piano",         "alignment": "muqt", "title": "Piano"},
    {"key": "mood",          "alignment": "muqt", "title": "Mood"},
    {"key": "tempo",         "alignment": "muqt", "title": "Tempo"},
    {"key": "vocal_gender", "alignment": "clap", "title": "Vocal"},
]

# Style shared across sources: keyed by semantic role.
ROLE_STYLES = {
    "all":     dict(color="#3A86FF", linestyle="-", label="all"),
    "loc":     dict(color="#2A9D8F", linestyle="-", label="localized"),
    "no_loc":  dict(color="#E63946", linestyle="-", label="ablated"),
}
ROLE_ORDER = ["all", "no_loc", "loc"]

# Map semantic role -> on-disk layer dir (same naming for both sources after the cleanup).
ROLE_TO_LAYER = {"all": "all", "loc": "loc", "no_loc": "ablated"}


def _acestep_path(concept: str, role: str) -> str:
    return f"{ACESTEP_ROOT}/concept_{concept}/{ROLE_TO_LAYER[role]}/caa"


def _audioldm_path(concept: str, role: str) -> str:
    return f"{AUDIOLDM_ROOT}/{concept}/{ROLE_TO_LAYER[role]}"


SOURCES = {
    "acestep":   {"label": "AceStep",   "path_for": _acestep_path},
    "audioldm2": {"label": "AudioLDM2", "path_for": _audioldm_path},
}

OUT_DIR = Path("steering/outputs/plots")


def load_run(run_dir: str, alignment_metric: str) -> pd.DataFrame | None:
    results_dir = Path(run_dir) / "protocol_results"
    align_csv = results_dir / f"{alignment_metric}.csv"
    lpaps_csv = results_dir / "lpaps.csv"
    if not align_csv.exists() or not lpaps_csv.exists():
        return None
    lpaps = pd.read_csv(lpaps_csv)[["alpha", "mean", "std"]].rename(
        columns={"mean": "lpaps_mean", "std": "lpaps_std"}
    )
    align = pd.read_csv(align_csv)[["alpha", "mean", "std"]].rename(
        columns={"mean": "align_mean", "std": "align_std"}
    )
    return lpaps.merge(align, on="alpha").sort_values("alpha").reset_index(drop=True)


def load_source(source_name: str) -> dict:
    """Return {(concept_key, role): df_pos_alphas} for the given source."""
    path_for = SOURCES[source_name]["path_for"]
    data = {}
    for concept in CONCEPTS:
        c_key = concept["key"]
        metric = concept["alignment"]
        for role in ROLE_ORDER:
            run_dir = path_for(c_key, role)
            df = load_run(run_dir, metric)
            if df is None:
                print(f"  SKIP {source_name} {c_key}/{role}: missing CSVs at {run_dir}")
                continue
            data[(c_key, role)] = df[df["alpha"] >= 0].reset_index(drop=True)
    return data


# =========================================================================
# Poster style
# =========================================================================
plt.rcParams.update({
    "font.size": 54,
    "axes.titlesize": 56,
    "axes.labelsize": 52,
    "xtick.labelsize": 0,
    "ytick.labelsize": 0,
    "legend.fontsize": 55,
    "lines.linewidth": 7,
    "lines.markersize": 20,
    "lines.markeredgewidth": 4,
})


def plot_raw(data: dict, source_label: str, out_path: Path) -> None:
    """Figure 1: raw alignment vs LPAPS, 1x4."""
    fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(48, 11))

    for col_idx, concept in enumerate(CONCEPTS):
        ax = axes[col_idx]
        c_key = concept["key"]
        metric_label = concept["alignment"].upper()

        max_lpaps_per_role = [
            data[(c_key, r)]["lpaps_mean"].max()
            for r in ROLE_ORDER if (c_key, r) in data
        ]
        if not max_lpaps_per_role:
            ax.set_axis_off()
            continue
        lpaps_cutoff = min(max_lpaps_per_role)

        for role in ROLE_ORDER:
            key = (c_key, role)
            if key not in data:
                continue
            df = data[key]
            df = df[df["lpaps_mean"] <= lpaps_cutoff]
            style = ROLE_STYLES[role]
            label = style["label"] if col_idx == 0 else None
            ax.plot(
                df["align_mean"], df["lpaps_mean"],
                linestyle=style["linestyle"],
                marker=".",
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=label, zorder=5,
            )

        if col_idx == len(CONCEPTS) // 2 - 1 or col_idx == len(CONCEPTS) // 2:
            ax.set_xlabel(f"{metric_label} ↑")
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("LPAPS ↓")
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.grid(True, alpha=0.15, linewidth=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ROLE_ORDER),
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle(source_label, y=1.16, fontsize=64)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved: {out_path}")


def plot_preservation(data: dict, source_label: str, out_path: Path) -> None:
    """Figure 2: delta-alignment vs preserved-LPAPS, 1x4."""
    fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(48, 11))

    for col_idx, concept in enumerate(CONCEPTS):
        ax = axes[col_idx]
        c_key = concept["key"]
        metric_label = concept["alignment"].upper()

        max_lpaps_per_role = [
            data[(c_key, r)]["lpaps_mean"].max()
            for r in ROLE_ORDER if (c_key, r) in data
        ]
        if not max_lpaps_per_role:
            ax.set_axis_off()
            continue
        lpaps_cutoff = min(max_lpaps_per_role)

        for role in ROLE_ORDER:
            key = (c_key, role)
            if key not in data:
                continue
            df = data[key]
            zero = df[df["alpha"] == 0]
            if zero.empty:
                continue
            align_baseline = zero["align_mean"].values[0]

            df = df[df["lpaps_mean"] <= lpaps_cutoff]
            delta_align = df["align_mean"] - align_baseline
            preserved = lpaps_cutoff - df["lpaps_mean"]

            style = ROLE_STYLES[role]
            label = style["label"] if col_idx == 0 else None
            ax.plot(
                delta_align, preserved,
                linestyle=style["linestyle"],
                marker=".",
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=label, zorder=5,
            )

        if col_idx == len(CONCEPTS) // 2 - 1 or col_idx == len(CONCEPTS) // 2:
            ax.set_xlabel(f"Δ{metric_label} (concept gain) →")
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Audio preservation →")
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.grid(True, alpha=0.15, linewidth=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ROLE_ORDER),
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(source_label, y=1.13, fontsize=64)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved: {out_path}")


def plot_combined_preservation(
    audioldm_data: dict,
    acestep_data: dict,
    out_path: Path,
) -> None:
    """2x4 figure: AudioLDM2 (top row), ACE-Step (bottom row).
    Cols: piano, tempo, mood, vocal_gender (positive direction, preservation curve).
    Single x/y axis labels; row labels (model names) at the left of each row.
    """
    col_specs = [
        ("piano",         "+ piano"),
        ("tempo",         "+ tempo"),
        ("mood",          "+ mood"),
        ("vocal_gender", "+ female vocal"),
    ]
    rows = [
        ("AudioLDM2", audioldm_data),
        ("ACE-Step",  acestep_data),
    ]

    fig, axes = plt.subplots(2, len(col_specs), figsize=(40, 18))

    # Pass 1: gather per-cell curve data without drawing yet.
    cell_curves: dict = {}
    cell_meta: dict = {}

    for row_idx, (model_name, data) in enumerate(rows):
        for col_idx, (c_key, col_title) in enumerate(col_specs):
            max_lpaps_per_role = [
                data[(c_key, r)]["lpaps_mean"].max()
                for r in ROLE_ORDER if (c_key, r) in data
            ]
            if not max_lpaps_per_role:
                axes[row_idx, col_idx].set_axis_off()
                continue
            lpaps_cutoff = min(max_lpaps_per_role)
            cell_meta[(row_idx, col_idx)] = (model_name, c_key, col_title)

            curves = []
            for role in ROLE_ORDER:
                key = (c_key, role)
                if key not in data:
                    continue
                df = data[key]
                zero = df[df["alpha"] == 0]
                if zero.empty:
                    continue
                align_baseline = zero["align_mean"].values[0]
                df = df[df["lpaps_mean"] <= lpaps_cutoff]
                delta_align = (df["align_mean"] - align_baseline).to_numpy()
                preserved = (lpaps_cutoff - df["lpaps_mean"]).to_numpy()
                curves.append((role, delta_align, preserved))
            cell_curves[(row_idx, col_idx)] = curves

    # Per-cell floor: highest of the curves' lowest preservations *within* the
    # cell, so every curve in that cell gets cut to the level of its
    # shortest-reaching neighbor.
    cell_floors: dict = {}
    for key, curves in cell_curves.items():
        mins = [p.min() for _, _, p in curves if len(p) > 0]
        if mins:
            cell_floors[key] = max(mins)

    # Pass 2: draw, masking each curve below the per-cell floor.
    active_axes: list = []
    for (row_idx, col_idx), curves in cell_curves.items():
        ax = axes[row_idx, col_idx]
        active_axes.append(ax)
        model_name, _, col_title = cell_meta[(row_idx, col_idx)]
        floor = cell_floors.get((row_idx, col_idx), 0.0)

        for role, delta_align, preserved in curves:
            mask = preserved >= floor
            if not mask.any():
                continue
            da_plot = delta_align[mask]
            pr_plot = preserved[mask]
            # Interpolate so the curve ends exactly at the cell floor (curves
            # are monotonic in alpha: preserved decreases as alpha grows).
            last_idx = int(np.flatnonzero(mask).max())
            if last_idx + 1 < len(preserved):
                p_hi, p_lo = preserved[last_idx], preserved[last_idx + 1]
                d_hi, d_lo = delta_align[last_idx], delta_align[last_idx + 1]
                t = (p_hi - floor) / (p_hi - p_lo)
                da_plot = np.append(da_plot, d_hi + t * (d_lo - d_hi))
                pr_plot = np.append(pr_plot, floor)

            style = ROLE_STYLES[role]
            # Legend only emitted from one panel so we get a single shared legend.
            label = style["label"] if (row_idx == 0 and col_idx == 0) else None
            ax.plot(
                da_plot, pr_plot,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=".",
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=label, zorder=5,
            )

        # vertical line at alpha=0 (Δalign = 0) plus a rotated "α=0" label sitting on the line.
        # ACE-Step curves crowd the middle, so push the label toward 3/4 of the cell height there.
        ax.axvline(0, color="gray", linestyle=":", linewidth=2, alpha=0.6, zorder=1)
        y_alpha = 0.25 if model_name == "ACE-Step" else 0.5
        ax.text(
            0, y_alpha, r"$\alpha=0$",
            transform=ax.get_xaxis_transform(),
            ha="center", va="center",
            rotation=90,
            fontsize=22, color="gray",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
            zorder=2,
        )

        if row_idx == 0:
            ax.set_title(col_title)
        if col_idx == len(col_specs) - 1:
            ax.set_ylabel(model_name, rotation=270, labelpad=60, fontsize=62)
            ax.yaxis.set_label_position("right")

    # Final styling pass.
    for ax in active_axes:
        ax.tick_params(
            axis="both", which="major",
            labelsize=44, length=18, width=4, pad=6,
            color="black",
        )
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        ax.set_xticks([x_lo + (x_hi - x_lo) * f for f in (0.2, 0.5, 0.8)])
        ax.set_yticks([y_lo + (y_hi - y_lo) * f for f in (0.25, 0.5, 0.75)])
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%+.2f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.15, linewidth=1)

    # Reserve outer margins for the supxlabel / supylabel. tight_layout's rect=
    # forces the grid into the inner box [left, bottom, right, top] of the figure.
    plt.tight_layout(rect=[0.07, 0.09, 1.0, 0.95])

    # Compute the grid's bounding box (in figure coords) so xlabel/ylabel center on
    # the grid of cells rather than on the whole figure.
    bboxes = [ax.get_position() for ax in axes.flat]
    grid_x0 = min(b.x0 for b in bboxes); grid_x1 = max(b.x1 for b in bboxes)
    grid_y0 = min(b.y0 for b in bboxes); grid_y1 = max(b.y1 for b in bboxes)
    grid_xc = (grid_x0 + grid_x1) / 2
    grid_yc = (grid_y0 + grid_y1) / 2

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Inline title: render "Layers:" as a normal entry on the same row, with no
        # visible handle (transparent line + zero handle length).
        from matplotlib.lines import Line2D
        title_proxy = Line2D([], [], color="none", marker="")
        leg = fig.legend(
            [title_proxy] + handles,
            ["Layers:"] + list(labels),
            loc="upper center",
            ncol=len(handles) + 1,
            bbox_to_anchor=(0.48, 1.06),
            frameon=True,
            handlelength=2.0,
            columnspacing=2.5,
            fontsize=65,
        )
        # Suppress the empty handle slot for the title entry so "Layers:" sits flush.
        first_handle = leg.legend_handles[0]
        first_handle.set_visible(False)
        # Bold the "localized" label.
        for txt in leg.get_texts():
            if txt.get_text() == "localized":
                txt.set_fontweight("bold")

    fig.supxlabel(r"Concept Score$\rightarrow$", fontsize=85, x=grid_xc, y=0.005)
    fig.supylabel(r"Audio Preservation $\rightarrow$", fontsize=85, x=0.04, y=grid_yc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved: {out_path}")


def plot_single_row_preservation(
    model_groups: list,
    out_path: Path,
    col_specs: list | None = None,
) -> None:
    """1xN preservation plot for one or more models laid out side-by-side.
    `model_groups` is a list of (model_name, data) tuples; columns are
    concatenated and each model's block is labeled above its columns.
    Mirrors plot_combined_preservation styling (per-cell floor cut + interp,
    tick locations, formatters, frame-on legend, supxlabel/supylabel).
    """
    if col_specs is None:
        col_specs = [
            ("piano",         "+ piano"),
            ("tempo",         "+ tempo"),
            ("vocal_gender", "+ female vocal"),
        ]

    n_per_model = len(col_specs)
    n_total = n_per_model * len(model_groups)
    # Figsize + margins chosen so each cell ends up roughly square.
    # inner grid ~46x6.5 in; 6 cells with wspace=0.12 => ~7x6.5 each.
    fig, axes = plt.subplots(
        1, n_total, figsize=(8 * n_total, 11),
        gridspec_kw={"wspace": 0.4},
    )
    if n_total == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    # Flat list of (panel_idx, model_idx, col_idx, model_name, data, c_key, c_title).
    panel_info = []
    for m_idx, (m_name, m_data) in enumerate(model_groups):
        for c_idx, (c_key, c_title) in enumerate(col_specs):
            panel_idx = m_idx * n_per_model + c_idx
            panel_info.append((panel_idx, m_idx, c_idx, m_name, m_data, c_key, c_title))

    # Pass 1: gather per-cell curve data.
    cell_curves: dict = {}
    for panel_idx, m_idx, c_idx, m_name, m_data, c_key, c_title in panel_info:
        max_lpaps_per_role = [
            m_data[(c_key, r)]["lpaps_mean"].max()
            for r in ROLE_ORDER if (c_key, r) in m_data
        ]
        if not max_lpaps_per_role:
            axes_list[panel_idx].set_axis_off()
            continue
        lpaps_cutoff = min(max_lpaps_per_role)

        curves = []
        for role in ROLE_ORDER:
            key = (c_key, role)
            if key not in m_data:
                continue
            df = m_data[key]
            zero = df[df["alpha"] == 0]
            if zero.empty:
                continue
            align_baseline = zero["align_mean"].values[0]
            df = df[df["lpaps_mean"] <= lpaps_cutoff]
            delta_align = (df["align_mean"] - align_baseline).to_numpy()
            preserved = (lpaps_cutoff - df["lpaps_mean"]).to_numpy()
            curves.append((role, delta_align, preserved))
        cell_curves[panel_idx] = curves

    # Per-cell floor: highest of curves' minima within the cell.
    cell_floors: dict = {}
    for panel_idx, curves in cell_curves.items():
        mins = [p.min() for _, _, p in curves if len(p) > 0]
        if mins:
            cell_floors[panel_idx] = max(mins)

    # Pass 2: draw, masking each curve below the per-cell floor.
    active_axes: list = []
    panel_info_by_idx = {p[0]: p for p in panel_info}
    for panel_idx, curves in cell_curves.items():
        ax = axes_list[panel_idx]
        active_axes.append(ax)
        _, _, _, _, _, _, c_title = panel_info_by_idx[panel_idx]
        floor = cell_floors.get(panel_idx, 0.0)

        for role, delta_align, preserved in curves:
            mask = preserved >= floor
            if not mask.any():
                continue
            da_plot = delta_align[mask]
            pr_plot = preserved[mask]
            last_idx = int(np.flatnonzero(mask).max())
            if last_idx + 1 < len(preserved):
                p_hi, p_lo = preserved[last_idx], preserved[last_idx + 1]
                d_hi, d_lo = delta_align[last_idx], delta_align[last_idx + 1]
                t = (p_hi - floor) / (p_hi - p_lo)
                da_plot = np.append(da_plot, d_hi + t * (d_lo - d_hi))
                pr_plot = np.append(pr_plot, floor)

            style = ROLE_STYLES[role]
            # Emit legend only from the very first panel.
            label = style["label"] if panel_idx == 0 else None
            ax.plot(
                pr_plot, da_plot,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=".",
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=label, zorder=5,
            )

        ax.axhline(0, color="gray", linestyle=":", linewidth=2, alpha=0.6, zorder=1)
        ax.text(
            0.5, 0, r"$\alpha=0$",
            transform=ax.get_yaxis_transform(),
            ha="center", va="center",
            rotation=0,
            fontsize=22, color="gray",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
            zorder=2,
        )
        ax.set_title(c_title, fontsize=50)

    # Final styling pass.
    for ax in active_axes:
        ax.set_box_aspect(1)
        ax.tick_params(
            axis="both", which="major",
            labelsize=50, length=18, width=4, pad=0,
            color="black",
        )
        ax.tick_params(axis="y", labelsize=42)
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        ax.set_xticks([x_lo + (x_hi - x_lo) * f for f in (1/10, 1/2, 9/10)])
        ax.set_yticks([y_lo + (y_hi - y_lo) * f for f in (0.25, 0.5, 0.75)])
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%+.2f"))
        ax.grid(True, alpha=0.15, linewidth=1)

    # Reserve top margin for the model-group labels; bottom for supxlabel.
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.78, bottom=0.18, wspace=0.4,
    )

    # Compute grid bbox for sup labels.
    bboxes = [ax.get_position() for ax in active_axes]
    grid_x0 = min(b.x0 for b in bboxes); grid_x1 = max(b.x1 for b in bboxes)
    grid_y0 = min(b.y0 for b in bboxes); grid_y1 = max(b.y1 for b in bboxes)
    grid_xc = (grid_x0 + grid_x1) / 2
    grid_yc = (grid_y0 + grid_y1) / 2

    # Per-model group labels above the column titles ("a)", "b)", ...).
    if len(model_groups) > 1:
        for m_idx, (m_name, _) in enumerate(model_groups):
            block_axes = axes_list[m_idx * n_per_model:(m_idx + 1) * n_per_model]
            block_bb = [a.get_position() for a in block_axes]
            x_center = (min(b.x0 for b in block_bb) + max(b.x1 for b in block_bb)) / 2
            top_y = max(b.y1 for b in block_bb) + 0.075
            label = f"{chr(ord('a') + m_idx)}) {m_name}"
            fig.text(x_center, top_y, label,
                     ha="center", va="bottom",
                     fontsize=70)

    handles, labels = active_axes[0].get_legend_handles_labels() if active_axes else ([], [])
    if handles:
        from matplotlib.lines import Line2D
        title_proxy = Line2D([], [], color="none", marker="")
        leg = fig.legend(
            [title_proxy] + handles,
            ["Steering layers:"] + list(labels),
            loc="upper center",
            ncol=len(handles) + 1,
            bbox_to_anchor=(0.5, 1.21),
            frameon=True,
            handlelength=2.0,
            columnspacing=2.5,
            fontsize=70,
        )
        first_handle = leg.legend_handles[0]
        first_handle.set_visible(False)
        for txt in leg.get_texts():
            if txt.get_text() == "localized":
                txt.set_fontweight("bold")

    fig.supxlabel(r"Preservation$\rightarrow$", fontsize=70, x=grid_xc, y=-0.005)
    fig.supylabel(r"Concept Gain$\rightarrow$", fontsize=70, x=0.005, y=grid_yc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.savefig(out_path, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    loaded = {}
    for source_name, source in SOURCES.items():
        print(f"\n=== {source_name} ===")
        data = load_source(source_name)
        if not data:
            print(f"  no data found for {source_name}; skipping")
            continue
        loaded[source_name] = data
        plot_raw(
            data, source["label"],
            OUT_DIR / f"caa_{source_name}_layers_raw.svg",
        )

    if "acestep" in loaded and "audioldm2" in loaded:
        plot_combined_preservation(
            loaded["audioldm2"], loaded["acestep"],
            OUT_DIR / "caa_layers_preservation.svg",
        )


if __name__ == "__main__":
    main()
