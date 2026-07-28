"""Plot previously computed exact change-count audit results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/change_frontier_blocks"))
    p.add_argument("--output", type=Path, default=Path("article/figures/fig_change_frontier.pdf"))
    args = p.parse_args()
    domains = [("bts", "Airline"), ("road", "Roadway"), ("divvy", "Shared mobility")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    for ax, (key, title) in zip(axes, domains):
        path = args.results / key / "exact_change_frontier.csv"
        frame = pd.read_csv(path)
        for block, group in frame.groupby("block_id", sort=True):
            ax.plot(group["changed_actions"], group["protected_improvement"], marker="o", linewidth=1.1, markersize=2.8, label=str(block))
        ax.axhline(0.0, color="0.35", linewidth=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Changed actions $c$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Protected improvement $Q_c$", fontsize=8)
    axes[-1].legend(fontsize=6, frameon=False, loc="best")
    fig.tight_layout(pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
