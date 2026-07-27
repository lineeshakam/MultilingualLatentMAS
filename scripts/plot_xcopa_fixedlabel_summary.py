#!/usr/bin/env python
"""Plot fixed-label XCOPA accuracy and IFL summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA = [
    {"lang": "et", "n": 500, "single_acc": 0.500, "latent_acc": 0.672, "text_acc": 0.520, "text_ifl": 0.316, "latent_ifl": 0.168},
    {"lang": "ht", "n": 500, "single_acc": 0.440, "latent_acc": 0.578, "text_acc": 0.512, "text_ifl": 0.414, "latent_ifl": 0.145},
    {"lang": "id", "n": 500, "single_acc": 0.736, "latent_acc": 0.838, "text_acc": 0.702, "text_ifl": 0.128, "latent_ifl": 0.179},
    {"lang": "it", "n": 500, "single_acc": 0.708, "latent_acc": 0.856, "text_acc": 0.684, "text_ifl": 0.102, "latent_ifl": 0.172},
    {"lang": "qu", "n": 500, "single_acc": 0.268, "latent_acc": 0.520, "text_acc": 0.500, "text_ifl": 0.388, "latent_ifl": 0.075},
    {"lang": "sw", "n": 500, "single_acc": 0.414, "latent_acc": 0.576, "text_acc": 0.500, "text_ifl": 0.430, "latent_ifl": 0.082},
    {"lang": "ta", "n": 500, "single_acc": 0.602, "latent_acc": 0.778, "text_acc": 0.596, "text_ifl": 0.166, "latent_ifl": 0.199},
    {"lang": "th", "n": 500, "single_acc": 0.664, "latent_acc": 0.814, "text_acc": 0.704, "text_ifl": 0.120, "latent_ifl": 0.175},
    {"lang": "tr", "n": 500, "single_acc": 0.646, "latent_acc": 0.786, "text_acc": 0.622, "text_ifl": 0.180, "latent_ifl": 0.189},
    {"lang": "vi", "n": 500, "single_acc": 0.714, "latent_acc": 0.838, "text_acc": 0.704, "text_ifl": 0.106, "latent_ifl": 0.179},
    {"lang": "zh", "n": 500, "single_acc": 0.792, "latent_acc": 0.888, "text_acc": 0.806, "text_ifl": 0.088, "latent_ifl": 0.111},
]


def main() -> None:
    out_dir = Path("results/xcopa_planner_solver_critic/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(DATA)
    macro = {
        "lang": "Macro",
        "n": 500,
        "single_acc": df["single_acc"].mean(),
        "latent_acc": df["latent_acc"].mean(),
        "text_acc": df["text_acc"].mean(),
        "text_ifl": df["text_ifl"].mean(),
        "latent_ifl": df["latent_ifl"].mean(),
    }
    plot_df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    x = range(len(plot_df))
    width = 0.25
    ax.bar([i - width for i in x], plot_df["single_acc"], width=width, label="Single Solver")
    ax.bar(x, plot_df["latent_acc"], width=width, label="Latent Planner-Solver-Critic")
    ax.bar([i + width for i in x], plot_df["text_acc"], width=width, label="Text Planner-Solver-Critic")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["lang"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Language")
    ax.set_title("XCOPA Accuracy by Language and Condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "xcopa_accuracy_by_condition.png", dpi=300)
    fig.savefig(out_dir / "xcopa_accuracy_by_condition.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    width = 0.34
    ax.bar([i - width / 2 for i in x], plot_df["text_ifl"], width=width, label="Text MAS IFL")
    ax.bar([i + width / 2 for i in x], plot_df["latent_ifl"], width=width, label="Latent MAS IFL")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["lang"])
    ax.set_ylim(0, max(plot_df["text_ifl"].max(), plot_df["latent_ifl"].max()) + 0.08)
    ax.set_ylabel("Involuntary Fidelity Loss")
    ax.set_xlabel("Language")
    ax.set_title("XCOPA Involuntary Fidelity Loss by Language")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "xcopa_ifl_by_condition.png", dpi=300)
    fig.savefig(out_dir / "xcopa_ifl_by_condition.pdf")
    plt.close(fig)

    csv_path = out_dir / "xcopa_fixedlabel_summary_used_for_plots.csv"
    plot_df.to_csv(csv_path, index=False)

    print(f"[plot] {out_dir / 'xcopa_accuracy_by_condition.png'}")
    print(f"[plot] {out_dir / 'xcopa_ifl_by_condition.png'}")
    print(f"[data] {csv_path}")


if __name__ == "__main__":
    main()
