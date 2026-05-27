"""2x4 preservation plot: AudioLDM2 (top) + ACE-Step (bottom).

Cols: piano / tempo / mood / vocal_gender.
x = preservation = LPAPS_cutoff - LPAPS.   y = Δalignment (concept gain).

Styled to match plot_caa_layers_preservation_1xN.py (same fontsizes, formatters,
sup labels, legend, square cells, no tight crop).

Per-cell floor: each panel is cut to the level of its shortest-reaching curve
(after the cutoff), and curves are interpolated to end exactly on that floor.

Output:
  steering/outputs/plots/caa_layers_preservation_2x4.svg

Run from repo root:
  python steering/eval/plot_caa_layers_preservation_2x4.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from caa_layers_common import (
    ROLE_STYLES, ROLE_ORDER, OUT_DIR,
    load_source, apply_poster_style,
)

apply_poster_style()


def plot_combined_preservation(
    audioldm_data: dict,
    acestep_data: dict,
    out_path: Path,
) -> None:
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

    fig, axes = plt.subplots(
        2, len(col_specs), figsize=(10 * len(col_specs), 20),
        gridspec_kw={"wspace": 0.35, "hspace": 0.01},
    )

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

    # Per-cell floor: highest of curves' minima within the cell.
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
                pr_plot, da_plot,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=".",
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=label, zorder=5,
            )

        # Horizontal line at alpha=0 (Δalign = 0) + a horizontal "α=0" label on the line.
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

        if row_idx == 0:
            ax.set_title(col_title, fontsize=50)
        if col_idx == len(col_specs) - 1:
            ax.set_ylabel(model_name, rotation=270, labelpad=60, fontsize=70)
            ax.yaxis.set_label_position("right")

    # Final styling pass (matches 1xN).
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

    # Reserve top for legend, bottom for supxlabel.
    fig.subplots_adjust(
        left=0.08, right=0.95, top=0.88, bottom=0.06, wspace=0.4, hspace=0.4,
    )

    # Center sup labels on the cell grid (not the full figure).
    bboxes = [ax.get_position() for ax in axes.flat]
    grid_x0 = min(b.x0 for b in bboxes); grid_x1 = max(b.x1 for b in bboxes)
    grid_y0 = min(b.y0 for b in bboxes); grid_y1 = max(b.y1 for b in bboxes)
    grid_xc = (grid_x0 + grid_x1) / 2
    grid_yc = (grid_y0 + grid_y1) / 2

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        title_proxy = Line2D([], [], color="none", marker="")
        leg = fig.legend(
            [title_proxy] + handles,
            ["Steering layers:"] + list(labels),
            loc="upper center",
            ncol=len(handles) + 1,
            bbox_to_anchor=(0.5, 1.01),
            frameon=True,
            handlelength=2.0,
            columnspacing=2.5,
            fontsize=70,
        )
        leg.legend_handles[0].set_visible(False)
        for txt in leg.get_texts():
            if txt.get_text() == "localized":
                txt.set_fontweight("bold")

    fig.supxlabel(r"Preservation$\rightarrow$", fontsize=70, x=grid_xc, y=0.00)
    fig.supylabel(r"Concept Gain$\rightarrow$", fontsize=70, x=-0.0, y=grid_yc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, transparent=False)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    audioldm = load_source("audioldm2")
    acestep  = load_source("acestep")
    if not audioldm or not acestep:
        print("  missing data for one or both sources; aborting")
        return
    plot_combined_preservation(
        audioldm, acestep,
        OUT_DIR / "caa_layers_preservation_2x4.svg",
    )


if __name__ == "__main__":
    main()
