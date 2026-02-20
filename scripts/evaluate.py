#!/usr/bin/env python3
"""Evaluate trained 3DGS models with PSNR, SSIM, and LPIPS metrics."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import math
from pathlib import Path

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

from src.gaussian_depth.data.colmap import Dataset, Parser
from src.gaussian_depth.utils.config import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(path, device="cuda"):
    """Load a trained Gaussian checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    splats = torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.to(device)) for k, v in ckpt["splats"].items()
    })
    return splats, ckpt.get("config", {})


@torch.no_grad()
def render_image(splats, camtoworld, K, width, height, sh_degree=3, config=None):
    """Render a single image from Gaussians."""
    viewmat = torch.linalg.inv(camtoworld[None])

    near = 0.01
    far = 1e10
    if config:
        near = config.get("gaussians", {}).get("near_plane", 0.01)
        far = config.get("gaussians", {}).get("far_plane", 1e10)

    renders, alphas, meta = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
        viewmats=viewmat,
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=True,
        near_plane=near,
        far_plane=far,
        render_mode="RGB+D",
    )

    color = renders[0, ..., :3].clamp(0, 1)  # [H, W, 3]
    depth = renders[0, ..., 3]                # [H, W]
    return color, depth


def evaluate_model(splats, valset, config, device, lpips_fn=None):
    """Evaluate model on validation set."""
    sh_degree = config.get("gaussians", {}).get("sh_degree", 3)

    psnrs = []
    ssims = []
    lpips_vals = []
    rendered_images = []

    for i in tqdm(range(len(valset)), desc="Evaluating"):
        data = valset[i]
        gt_pixels = data["image"].float().to(device) / 255.0
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        height, width = gt_pixels.shape[:2]

        rendered, depth = render_image(
            splats, camtoworld, K, width, height, sh_degree, config,
        )

        # PSNR
        mse = F.mse_loss(rendered, gt_pixels)
        psnr = -10 * math.log10(max(mse.item(), 1e-10))
        psnrs.append(psnr)

        # SSIM (on CPU numpy)
        rendered_np = rendered.cpu().numpy()
        gt_np = gt_pixels.cpu().numpy()
        ssim_val = ssim_fn(rendered_np, gt_np, channel_axis=2, data_range=1.0)
        ssims.append(ssim_val)

        # LPIPS
        if lpips_fn is not None:
            r_tensor = rendered.permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [1,3,H,W] in [-1,1]
            g_tensor = gt_pixels.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            lpips_val = lpips_fn(r_tensor, g_tensor).item()
            lpips_vals.append(lpips_val)

        # Store first 5 rendered images for visualization
        if i < 5:
            rendered_images.append({
                "rendered": (rendered.cpu().numpy() * 255).astype(np.uint8),
                "gt": (gt_pixels.cpu().numpy() * 255).astype(np.uint8),
                "depth": depth.cpu().numpy(),
            })

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
        "psnr_std": float(np.std(psnrs)),
        "ssim_std": float(np.std(ssims)),
        "n_images": len(valset),
        "n_gaussians": splats["means"].shape[0],
        "rendered_images": rendered_images,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths to evaluate (label:path format)")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    data_cfg = config["data"]
    colmap_parser = Parser(
        data_dir=data_cfg["data_dir"],
        factor=data_cfg["data_factor"],
        normalize=data_cfg["normalize"],
        test_every=data_cfg["test_every"],
    )
    valset = Dataset(colmap_parser, split="test")
    logger.info(f"Validation set: {len(valset)} images")

    # LPIPS model
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ckpt_spec in args.checkpoints:
        if ":" in ckpt_spec:
            label, ckpt_path = ckpt_spec.split(":", 1)
        else:
            label = Path(ckpt_spec).parent.name
            ckpt_path = ckpt_spec

        logger.info(f"\nEvaluating: {label} ({ckpt_path})")
        splats, ckpt_config = load_checkpoint(ckpt_path, device)
        eval_config = ckpt_config if ckpt_config else config

        metrics = evaluate_model(splats, valset, eval_config, device, lpips_fn)

        logger.info(f"  PSNR:  {metrics['psnr']:.2f} +/- {metrics['psnr_std']:.2f} dB")
        logger.info(f"  SSIM:  {metrics['ssim']:.4f} +/- {metrics['ssim_std']:.4f}")
        if metrics["lpips"] is not None:
            logger.info(f"  LPIPS: {metrics['lpips']:.4f}")
        logger.info(f"  Gaussians: {metrics['n_gaussians']:,}")

        # Save rendered images
        render_dir = results_dir / f"{label}_renders"
        render_dir.mkdir(exist_ok=True)
        for j, img_data in enumerate(metrics["rendered_images"]):
            imageio.imwrite(str(render_dir / f"rendered_{j:03d}.png"), img_data["rendered"])
            imageio.imwrite(str(render_dir / f"gt_{j:03d}.png"), img_data["gt"])

        # Store results without images
        result_clean = {k: v for k, v in metrics.items() if k != "rendered_images"}
        all_results[label] = result_clean

        # Save per-model result
        with open(results_dir / f"{label}_eval.json", "w") as f:
            json.dump(result_clean, f, indent=2)

        del splats
        torch.cuda.empty_cache()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<15} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'#Gaussians':>12}")
    logger.info("-" * 55)
    for label, r in all_results.items():
        lpips_str = f"{r['lpips']:.4f}" if r['lpips'] is not None else "N/A"
        logger.info(f"{label:<15} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {lpips_str:>8} {r['n_gaussians']:>12,}")

    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSummary saved to {results_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
