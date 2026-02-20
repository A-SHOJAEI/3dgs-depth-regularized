#!/usr/bin/env python3
"""Generate comparison visualizations from evaluation results."""

import sys
sys.path.insert(0, ".")

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {"baseline": "#636EFA", "depth_reg": "#00CC96", "mcmc": "#EF553B"}
LABELS = {"baseline": "Baseline", "depth_reg": "Depth-Reg", "mcmc": "MCMC"}


def plot_metrics_comparison(summary, results_dir):
    """Bar charts comparing PSNR, SSIM, LPIPS across models."""
    models = list(summary.keys())
    labels = [LABELS.get(m, m) for m in models]
    colors = [COLORS.get(m, "#999") for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # PSNR
    psnrs = [summary[m]["psnr"] for m in models]
    bars = axes[0].bar(labels, psnrs, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR (Higher is Better)", fontweight="bold")
    axes[0].set_ylim(min(psnrs) - 1, max(psnrs) + 1)
    for bar, val in zip(bars, psnrs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    # SSIM
    ssims = [summary[m]["ssim"] for m in models]
    bars = axes[1].bar(labels, ssims, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM (Higher is Better)", fontweight="bold")
    axes[1].set_ylim(min(ssims) - 0.02, min(max(ssims) + 0.02, 1.0))
    for bar, val in zip(bars, ssims):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    # LPIPS
    lpips_vals = [summary[m].get("lpips", 0) or 0 for m in models]
    bars = axes[2].bar(labels, lpips_vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[2].set_ylabel("LPIPS")
    axes[2].set_title("LPIPS (Lower is Better)", fontweight="bold")
    axes[2].set_ylim(0, max(lpips_vals) * 1.3 + 0.01)
    for bar, val in zip(bars, lpips_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    path = results_dir / "metrics_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_training_curves(results_dir):
    """Plot training loss curves from training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ckpt_dir in sorted(results_dir.parent.glob("checkpoints/*")):
        metrics_path = ckpt_dir / "training_metrics.json"
        if not metrics_path.exists():
            continue

        label = ckpt_dir.name.rsplit("_", 2)[0]  # Remove timestamp
        color = COLORS.get(label, "#999")
        display = LABELS.get(label, label)

        with open(metrics_path) as f:
            metrics = json.load(f)

        steps = [m["step"] for m in metrics]
        losses = [m["total"] for m in metrics]
        n_gs = [m["n_gaussians"] for m in metrics]

        axes[0].plot(steps, losses, label=display, color=color, linewidth=2)
        axes[1].plot(steps, [g/1000 for g in n_gs], label=display, color=color, linewidth=2)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gaussians (K)")
    axes[1].set_title("Number of Gaussians", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = results_dir / "training_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_render_comparison(results_dir):
    """Create side-by-side comparison of rendered vs GT images."""
    import imageio.v2 as imageio

    model_dirs = sorted(results_dir.glob("*_renders"))
    if not model_dirs:
        logger.info("No render directories found, skipping render comparison.")
        return

    n_models = len(model_dirs)
    n_images = min(3, len(list(model_dirs[0].glob("rendered_*.png"))))

    fig, axes = plt.subplots(n_images, n_models + 1, figsize=(4 * (n_models + 1), 4 * n_images))
    if n_images == 1:
        axes = axes[None, :]

    for j, model_dir in enumerate(model_dirs):
        label = model_dir.name.replace("_renders", "")
        display = LABELS.get(label, label)

        for i in range(n_images):
            rendered = imageio.imread(str(model_dir / f"rendered_{i:03d}.png"))
            axes[i, j].imshow(rendered)
            axes[i, j].set_title(display if i == 0 else "", fontweight="bold")
            axes[i, j].axis("off")

            if j == 0:
                gt = imageio.imread(str(model_dir / f"gt_{i:03d}.png"))
                axes[i, -1].imshow(gt)
                axes[i, -1].set_title("Ground Truth" if i == 0 else "", fontweight="bold")
                axes[i, -1].axis("off")

    plt.tight_layout()
    path = results_dir / "render_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    results_dir = Path("results")

    summary_path = results_dir / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        plot_metrics_comparison(summary, results_dir)
    else:
        logger.warning(f"No evaluation summary found at {summary_path}")

    plot_training_curves(results_dir)
    plot_render_comparison(results_dir)

    logger.info("All visualizations generated.")


if __name__ == "__main__":
    main()
