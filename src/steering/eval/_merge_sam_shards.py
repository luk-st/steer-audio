"""Merge per-shard vocal_gender_sam_shard*.csv files into vocal_gender_sam.csv,
then compute correlations + plot vs CLAP / MuQ-T.

Usage:
    python steering/eval/_merge_sam_shards.py \
        --steering_dir pwr-mount/steering/outputs/ace_step/concept_vocal_gender/loc/sae
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from fire import Fire
from scipy.stats import pearsonr, spearmanr


def _correlate(alignment_df, ext_df, name, ext_col="mean"):
    ext = ext_df[["alpha", ext_col]].rename(columns={ext_col: "mean"})
    merged = alignment_df.merge(ext, on="alpha", suffixes=("_align", "_ext"))
    if len(merged) < 3:
        return {"metric": name, "n_alpha": len(merged)}
    pr = pearsonr(merged["mean_align"], merged["mean_ext"])
    sr = spearmanr(merged["mean_align"], merged["mean_ext"])
    return {
        "metric": name,
        "n_alpha": int(len(merged)),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def _plot(clap_df, muqt_df, vg_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, label, color in [
        (axes[0], clap_df, "CLAP (music)", "#009FB7"),
        (axes[1], muqt_df, "MuQ-T", "#FE4A49"),
    ]:
        ax.errorbar(df["alpha"], df["mean"], yerr=df["std"],
                    fmt="-o", capsize=4, color=color, linewidth=2, label=label)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(vg_df["alpha"], vg_df["mean"], "-s",
                 color="#6A4C93", linewidth=2, label="P(female | SAM-audio stem)")
        ax2.set_ylabel("P(female | SAM-audio stem)", color="#6A4C93")
        ax2.tick_params(axis="y", labelcolor="#6A4C93")
        ax2.set_ylim(0.0, 1.0)
    plt.suptitle("Female vocals: CLAP / MuQ-T vs SAM-audio + wav2vec2 gender")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main(steering_dir: str):
    output_dir = os.path.join(steering_dir, "protocol_results")
    shards = sorted(glob.glob(os.path.join(output_dir, "vocal_gender_sam_shard*.csv")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {output_dir}")
    print(f"Merging {len(shards)} shards: {[os.path.basename(s) for s in shards]}")
    parts = [pd.read_csv(s) for s in shards]
    vg_df = pd.concat(parts, ignore_index=True).sort_values("alpha").reset_index(drop=True)
    merged_path = os.path.join(output_dir, "vocal_gender_sam.csv")
    vg_df.to_csv(merged_path, index=False)
    print(f"Saved merged {merged_path} ({len(vg_df)} rows)")

    clap_df = pd.read_csv(os.path.join(output_dir, "clap.csv"))
    muqt_df = pd.read_csv(os.path.join(output_dir, "muqt.csv"))

    rows = [
        _correlate(clap_df, vg_df, "clap_vs_female_prob_sam"),
        _correlate(muqt_df, vg_df, "muqt_vs_female_prob_sam"),
        _correlate(clap_df, vg_df, "clap_vs_vocal_rms_sam", ext_col="vocal_rms_mean"),
        _correlate(muqt_df, vg_df, "muqt_vs_vocal_rms_sam", ext_col="vocal_rms_mean"),
    ]
    val_df = pd.DataFrame(rows)
    val_df.to_csv(os.path.join(output_dir, "external_validation_vocals_sam.csv"), index=False)
    print("\nSAM-audio validation correlations:")
    print(val_df.to_string(index=False))

    _plot(clap_df, muqt_df, vg_df,
          os.path.join(output_dir, "external_validation_vocals_sam.png"))


if __name__ == "__main__":
    Fire(main)
