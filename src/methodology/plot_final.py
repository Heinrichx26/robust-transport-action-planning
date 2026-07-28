import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot completed structural robust results.")
    parser.add_argument("--root", type=Path, default=Path("results/structural_robust_final/combined"))
    parser.add_argument("--output-dir", type=Path, default=Path("article/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(root / "main_results.csv")
    coverage = pd.read_csv(root / "coverage_summary.csv")
    labels = {"bts": "Airline", "road": "Roadway", "divvy": "Shared mobility"}
    colors = {"bts": "#31688e", "road": "#35b779", "divvy": "#440154"}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    for dataset in ("bts", "road", "divvy"):
        part = results[results.dataset == dataset]
        x = 100 * part.budget_fraction
        axes[0].plot(x, part.gain_over_loss_priority_pct, marker="o", label=labels[dataset], color=colors[dataset])
        axes[1].plot(x, part.gain_over_strongest_frontier_pct, marker="o", color=colors[dataset])
        cov = coverage[coverage.dataset == dataset]
        axes[2].plot(100 * cov.budget_fraction, 100 * cov.absolute_coverage, marker="o", color=colors[dataset])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[2].axhline(95, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Gain in worst-setting plan value (%)")
    axes[1].set_ylabel("Gain over largest adapted mean (%)")
    axes[2].set_ylabel("Adaptive coverage (%)")
    for ax in axes:
        ax.set_xlabel("Actions available (%)")
        ax.set_xticks([1, 3, 5, 10, 20])
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("(a) Compared with loss priority")
    axes[1].set_title("(b) Compared with recent mechanisms")
    axes[2].set_title("(c) Sequential complete-plan bound")
    fig.tight_layout()
    fig.savefig(out / "fig_structural_robust_results.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_structural_robust_results.png", dpi=300, bbox_inches="tight")

    certificate = pd.read_csv(root / "action_certificate_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for dataset in ("bts", "road", "divvy"):
        part = certificate[certificate.dataset == dataset]
        x = 100 * part.budget_fraction
        axes[0].plot(x, part.mean_paired_margin, marker="o", label=labels[dataset], color=colors[dataset])
        axes[1].plot(x, part.one_sided_95_lower_limit, marker="o", color=colors[dataset])
        released = part[part.use_structural_plan > 0.5]
        axes[1].scatter(
            100 * released.budget_fraction,
            released.one_sided_95_lower_limit,
            s=44,
            color=colors[dataset],
            edgecolor="black",
            zorder=3,
        )
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Actions available (%)")
        ax.set_xticks([1, 3, 5, 10, 20])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Protected paired improvement")
    axes[1].set_ylabel("Earlier-period lower confidence limit")
    axes[0].set_title("(a) Later-block paired improvement")
    axes[1].set_title("(b) Capacity-level release check")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_action_certificate.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_action_certificate.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
