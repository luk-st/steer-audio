import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

BAR_WIDTH = 0.75 / 2
FIGSIZE = (12, 5)
PROMPT1_COLOR = "#731DD8"
PROMPT2_COLOR = "#D4B483"
HIGHLIGHT_PROMPT1_COLOR = "#5D00C7"
HIGHLIGHT_PROMPT2_COLOR = "#E29522"


def plot_tracing_results(
    dataframe: pd.DataFrame,
    excluded_layers: list[str] = [],
    highlight_layers: list[str] = [],
    left_reference_idx: int | None = None,
    right_reference_idx: int | None = None,
    out_file_path: str | None = None,
    yticks: list[float] = [0.05, 0.10, 0.15, 0.20],
) -> None:
    layers = dataframe["Block"].tolist()
    layers = [layer for layer in layers if layer not in excluded_layers]
    layers_index = [i for i in range(len(layers))]
    p1_sims = [dataframe[dataframe["Block"] == layer]["clap_1_mean"].values[0] for layer in layers]
    p2_sims = [dataframe[dataframe["Block"] == layer]["clap_2_mean"].values[0] for layer in layers]
    prompt_p1 = dataframe["clap_prompt_1"].values[0]
    prompt_p2 = dataframe["clap_prompt_2"].values[0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(
        f'Clean feature = "{prompt_p1}", corrupted feature = "{prompt_p2}"', fontsize=15, fontweight="bold", pad=40
    )
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.spines.top"] = False
    ax.grid(axis="y", linestyle="--", color="grey", alpha=0.3)

    colors_p1 = [PROMPT1_COLOR if layer not in highlight_layers else HIGHLIGHT_PROMPT1_COLOR for layer in layers]
    colors_p2 = [PROMPT2_COLOR if layer not in highlight_layers else HIGHLIGHT_PROMPT2_COLOR for layer in layers]

    ax.bar(layers, p1_sims, BAR_WIDTH, label=f'CLAP("{prompt_p1}")', color=colors_p1)
    ax.bar(
        [idx + BAR_WIDTH for idx in layers_index], p2_sims, BAR_WIDTH, label=f'CLAP("{prompt_p2}")', color=colors_p2
    )

    for highlight_layer in highlight_layers:
        idx = layers.index(highlight_layer)
        ax.text(
            idx,
            p1_sims[idx] + 0.005,
            f"{p1_sims[idx]:.2f}",
            ha="center",
            fontsize=12,
            color="black",
            fontweight="bold",
        )
        ax.text(
            idx + BAR_WIDTH,
            p2_sims[idx] + 0.005,
            f"{p2_sims[idx]:.2f}",
            ha="center",
            fontsize=12,
            color="black",
            fontweight="bold",
        )

    ax.set_xticks([i + BAR_WIDTH / 2 for i in layers_index])
    ax.set_xticklabels(layers, rotation=45, fontsize=15)

    for highlight_layer in highlight_layers:
        labels = ax.get_xticklabels()
        idx = layers.index(highlight_layer)
        labels[idx].set_fontweight("bold")  # Correcting this part
        ax.set_xticklabels(labels)

    if left_reference_idx is not None:
        ax.axvline(x=left_reference_idx + 1.8 * (BAR_WIDTH), color="black", linestyle="--", linewidth=1)
    if right_reference_idx is not None:
        ax.axvline(x=right_reference_idx - 0.8 * (BAR_WIDTH), color="black", linestyle="--", linewidth=1)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=15)
    ax.set_xlabel("Layers with clean prompt", fontsize=15)
    ax.set_xlim(-(BAR_WIDTH / 2), len(layers) - BAR_WIDTH)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=14)

    if out_file_path is not None:
        plt.savefig(out_file_path, bbox_inches="tight")
    else:
        plt.show()
