"""1xN preservation plot: models concatenated left-to-right as concept columns.

Default produces 1x6: AudioLDM2 (piano / tempo / vocal_gender) | ACE-Step (same).
x = preservation = LPAPS_cutoff - LPAPS.   y = Δalignment (concept gain).

Per-cell floor: each panel is cut to the level of its shortest-reaching curve
(after the cutoff), and curves are interpolated to end exactly on that floor.

Output:
  steering/outputs/plots/caa_layers_preservation_1xN.svg

Run from repo root:
  python steering/eval/plot_caa_layers_preservation_1xN.py
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


def plot_single_row_preservation(
    model_groups: list,
    out_path: Path,
    col_specs: list | None = None,
) -> None:
    """One row per `model_groups`; columns of each model's block come from `col_specs`."""
    if col_specs is None:
        col_specs = [
            ("piano",         "+ piano"),
            ("tempo",         "+ tempo"),
            ("vocal_gender", "+ female vocal"),
        ]

    n_per_model = len(col_specs)
    n_total = n_per_model * len(model_groups)
    fig, axes = plt.subplots(
        1, n_total, figsize=(8 * n_total, 12),
        gridspec_kw={"wspace": 0.4},
    )
    axes_list = [axes] if n_total == 1 else list(axes)

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
        left=0.07, right=0.99, top=0.77, bottom=0.1, wspace=0.4,
    )

    # Center sup labels on the cell grid.
    bboxes = [ax.get_position() for ax in active_axes]
    grid_x0 = min(b.x0 for b in bboxes); grid_x1 = max(b.x1 for b in bboxes)
    grid_y0 = min(b.y0 for b in bboxes); grid_y1 = max(b.y1 for b in bboxes)
    grid_xc = (grid_x0 + grid_x1) / 2
    grid_yc = (grid_y0 + grid_y1) / 2

    # Per-model group labels above each block of columns ("a)", "b)", ...).
    if len(model_groups) > 1:
        for m_idx, (m_name, _) in enumerate(model_groups):
            block_axes = axes_list[m_idx * n_per_model:(m_idx + 1) * n_per_model]
            block_bb = [a.get_position() for a in block_axes]
            x_center = (min(b.x0 for b in block_bb) + max(b.x1 for b in block_bb)) / 2
            top_y = max(b.y1 for b in block_bb) + 0.06
            label = f"{chr(ord('a') + m_idx)}) {m_name}"
            fig.text(x_center, top_y, label,
                     ha="center", va="bottom",
                     fontsize=70)

    handles, labels = active_axes[0].get_legend_handles_labels() if active_axes else ([], [])
    if handles:
        title_proxy = Line2D([], [], color="none", marker="")
        leg = fig.legend(
            [title_proxy] + handles,
            ["Steering layers:"] + list(labels),
            loc="upper center",
            ncol=len(handles) + 1,
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
            handlelength=2.0,
            columnspacing=2.5,
            fontsize=70,
        )
        leg.legend_handles[0].set_visible(False)
        for txt in leg.get_texts():
            if txt.get_text() == "localized":
                txt.set_fontweight("bold")

    fig.supxlabel(r"Preservation$\rightarrow$", fontsize=70, x=grid_xc, y=0.015)
    fig.supylabel(r"Concept Gain$\rightarrow$", fontsize=70, x=0.005, y=grid_yc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, transparent=True)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    audioldm = load_source("audioldm2")
    acestep  = load_source("acestep")
    if not audioldm or not acestep:
        print("  missing data for one or both sources; aborting")
        return
    plot_single_row_preservation(
        [("AudioLDM2", audioldm), ("ACE-Step", acestep)],
        OUT_DIR / "caa_layers_preservation_1xN.svg",
    )


if __name__ == "__main__":
    main()
