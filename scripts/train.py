#!/usr/bin/env python3
"""Train 3D Gaussian Splatting with optional depth regularization.

Supports three training modes:
1. Baseline (default strategy, no depth loss)
2. Depth-regularized (default strategy + monocular depth priors)
3. MCMC (Markov Chain Monte Carlo densification strategy)
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from src.gaussian_depth.data.colmap import Dataset, Parser
from src.gaussian_depth.utils.config import ensure_dirs, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def knn(points: torch.Tensor, k: int = 4) -> torch.Tensor:
    """K-nearest neighbor distances (batched for memory efficiency)."""
    N = points.shape[0]
    all_dists = []
    batch_size = min(4096, N)
    for i in range(0, N, batch_size):
        batch = points[i:i + batch_size]
        dists = torch.cdist(batch, points)  # [batch, N]
        topk_dists, _ = dists.topk(k, largest=False)
        all_dists.append(topk_dists)
    return torch.cat(all_dists, dim=0)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to zeroth-order SH coefficient."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def fused_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two images. Input: [B, C, H, W] in [0,1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu12

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim.mean()


def create_splats(points, rgbs, scene_scale, config):
    """Initialize Gaussian parameters from COLMAP SfM points."""
    N = points.shape[0]
    device = points.device

    # Compute initial scale from KNN distances
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * config["gaussians"]["init_scale"]).unsqueeze(-1).repeat(1, 3)

    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    opacities = torch.logit(torch.full((N,), config["gaussians"]["init_opacity"], device=device))

    sh_degree = config["gaussians"]["sh_degree"]
    K = (sh_degree + 1) ** 2
    colors = torch.zeros(N, K, 3, device=device)
    colors[:, 0, :] = rgb_to_sh(rgbs)

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
        "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
    }).to(device)

    return splats


def create_optimizers(splats, config, batch_size=1):
    """Create Adam optimizers for each Gaussian parameter group."""
    lr_cfg = config["lr"]
    BS = batch_size

    optimizers = {}
    for name, lr in [
        ("means", lr_cfg["means"]),
        ("scales", lr_cfg["scales"]),
        ("quats", lr_cfg["quats"]),
        ("opacities", lr_cfg["opacities"]),
        ("sh0", lr_cfg["sh0"]),
        ("shN", lr_cfg["shN"]),
    ]:
        optimizers[name] = torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
    return optimizers


def create_schedulers(optimizers, config):
    """Create exponential LR schedulers for means."""
    max_steps = config["training"]["max_steps"]
    schedulers = {}
    for name, opt in optimizers.items():
        if name == "means":
            schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=0.01 ** (1.0 / max_steps)
            )
        else:
            schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=1.0  # no decay
            )
    return schedulers


def create_strategy(config, scene_scale):
    """Create densification strategy."""
    strat_cfg = config["strategy"]

    if strat_cfg["type"] == "mcmc":
        strategy = MCMCStrategy(
            cap_max=strat_cfg["cap_max"],
            noise_lr=strat_cfg["noise_lr"],
            refine_start_iter=strat_cfg["mcmc_refine_start_iter"],
            refine_every=strat_cfg["mcmc_refine_every"],
        )
    else:
        strategy = DefaultStrategy(
            prune_opa=strat_cfg["prune_opa"],
            grow_grad2d=strat_cfg["grow_grad2d"],
            grow_scale3d=strat_cfg["grow_scale3d"],
            grow_scale2d=strat_cfg["grow_scale2d"],
            refine_start_iter=strat_cfg["refine_start_iter"],
            refine_stop_iter=strat_cfg["refine_stop_iter"],
            refine_every=strat_cfg["refine_every"],
            reset_every=strat_cfg["reset_every"],
            absgrad=strat_cfg["absgrad"],
        )

    if strat_cfg["type"] == "mcmc":
        state = strategy.initialize_state()
    else:
        state = strategy.initialize_state(scene_scale=scene_scale)
    return strategy, state


def generate_depth_priors(parser, config, device="cuda"):
    """Generate monocular depth maps using Depth Anything V2."""
    from transformers import pipeline

    depth_cfg = config["depth"]
    model_id = f"depth-anything/{depth_cfg['depth_model']}"
    logger.info(f"Loading depth model: {model_id}")

    pipe = pipeline("depth-estimation", model=model_id, device=device)

    depth_dir = Path(parser.data_dir) / "depths"
    depth_dir.mkdir(exist_ok=True)

    from PIL import Image as PILImage

    for i, img_path in enumerate(parser.image_paths):
        depth_path = depth_dir / (Path(img_path).stem + ".npy")
        if depth_path.exists():
            continue

        image = PILImage.open(img_path).convert("RGB")
        result = pipe(image)
        depth_map = np.array(result["depth"])

        # Resize to match downsampled image size
        camera_id = parser.camera_ids[i]
        target_w, target_h = parser.imsize_dict[camera_id]
        from PIL import Image as PILImage2
        depth_pil = PILImage2.fromarray(depth_map.astype(np.float32), mode="F")
        depth_resized = np.array(depth_pil.resize((target_w, target_h), PILImage2.BILINEAR))

        np.save(depth_path, depth_resized)

        if (i + 1) % 20 == 0:
            logger.info(f"  Depth maps: {i+1}/{len(parser.image_paths)}")

    logger.info(f"Depth maps saved to {depth_dir}")
    return depth_dir


def load_depth_map(depth_dir, image_name):
    """Load a precomputed depth map."""
    depth_path = depth_dir / (Path(image_name).stem + ".npy")
    if depth_path.exists():
        return torch.from_numpy(np.load(depth_path)).float()
    return None


def render_step(splats, camtoworld, K, width, height, sh_degree, config):
    """Render Gaussians for a single camera view."""
    viewmat = torch.linalg.inv(camtoworld[None])

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
        near_plane=config["gaussians"]["near_plane"],
        far_plane=config["gaussians"]["far_plane"],
        render_mode="RGB+D",
    )

    # renders shape: [1, H, W, 4] (RGB + depth)
    color = renders[..., :3]   # [1, H, W, 3]
    depth = renders[..., 3:4]  # [1, H, W, 1]

    return color, depth, alphas, meta


def compute_loss(rendered, gt_pixels, rendered_depth, depth_gt, config, scene_scale):
    """Compute training loss (L1 + SSIM + optional depth)."""
    train_cfg = config["training"]
    ssim_lambda = train_cfg["ssim_lambda"]

    # Photometric loss
    l1_loss = F.l1_loss(rendered, gt_pixels)

    # SSIM loss
    rendered_perm = rendered.permute(0, 3, 1, 2)  # [B, 3, H, W]
    gt_perm = gt_pixels.permute(0, 3, 1, 2)
    ssim_val = fused_ssim(rendered_perm, gt_perm)
    ssim_loss = 1.0 - ssim_val

    loss = (1.0 - ssim_lambda) * l1_loss + ssim_lambda * ssim_loss

    loss_dict = {"l1": l1_loss.item(), "ssim": ssim_val.item(), "total": loss.item()}

    # Depth regularization
    if config["depth"]["enabled"] and depth_gt is not None:
        depth_lambda = config["depth"]["lambda"]
        # Loss in disparity space for better gradients
        rendered_d = rendered_depth.squeeze(-1).squeeze(0)  # [H, W]

        if config["depth"]["use_disparity"]:
            disp = torch.where(rendered_d > 0, 1.0 / rendered_d, torch.zeros_like(rendered_d))
            disp_gt = torch.where(depth_gt > 0, 1.0 / depth_gt, torch.zeros_like(depth_gt))
            depth_loss = F.l1_loss(disp, disp_gt) * scene_scale
        else:
            depth_loss = F.l1_loss(rendered_d, depth_gt) * scene_scale

        loss = loss + depth_lambda * depth_loss
        loss_dict["depth"] = depth_loss.item()
        loss_dict["total"] = loss.item()

    # Regularization
    if train_cfg["opacity_reg"] > 0:
        opa = torch.sigmoid(splats["opacities"])
        opa_reg = -(opa * torch.log(opa + 1e-10) + (1 - opa) * torch.log(1 - opa + 1e-10)).mean()
        loss = loss + train_cfg["opacity_reg"] * opa_reg

    return loss, loss_dict


def train(config, run_label, depth_enabled=False, strategy_type="default"):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Override config for this run
    config["depth"]["enabled"] = depth_enabled
    config["strategy"]["type"] = strategy_type

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"]["output_dir"]) / f"{run_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"3D Gaussian Splatting Training — {run_label}")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    data_cfg = config["data"]
    logger.info(f"Loading COLMAP data from {data_cfg['data_dir']}...")

    parser = Parser(
        data_dir=data_cfg["data_dir"],
        factor=data_cfg["data_factor"],
        normalize=data_cfg["normalize"],
        test_every=data_cfg["test_every"],
    )

    trainset = Dataset(parser, split="train", load_depths=depth_enabled)
    valset = Dataset(parser, split="test")

    scene_scale = parser.scene_scale * 1.1
    logger.info(f"  Train images: {len(trainset)}")
    logger.info(f"  Val images: {len(valset)}")
    logger.info(f"  SfM points: {parser.points.shape[0]}")
    logger.info(f"  Scene scale: {scene_scale:.4f}")

    # ── Generate depth priors ────────────────────────────────────
    depth_dir = None
    if depth_enabled:
        depth_dir = generate_depth_priors(parser, config, device=device)
        torch.cuda.empty_cache()

    # ── Initialize Gaussians ─────────────────────────────────────
    points = torch.from_numpy(parser.points).float().to(device)
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float().to(device)

    splats = create_splats(points, rgbs, scene_scale, config)
    logger.info(f"  Initial Gaussians: {splats['means'].shape[0]:,}")

    optimizers = create_optimizers(splats, config)
    schedulers = create_schedulers(optimizers, config)

    # ── Strategy ─────────────────────────────────────────────────
    strategy, strategy_state = create_strategy(config, scene_scale)
    logger.info(f"  Strategy: {strategy_type}")

    # ── Training loop ────────────────────────────────────────────
    max_steps = config["training"]["max_steps"]
    sh_degree = config["gaussians"]["sh_degree"]
    sh_interval = config["gaussians"]["sh_degree_interval"]
    eval_steps = config["evaluation"]["eval_steps"]
    save_steps = config["evaluation"]["save_steps"]

    metrics_log = []
    best_psnr = 0.0
    start_time = time.time()

    logger.info(f"Starting training for {max_steps} steps...")

    for step in range(max_steps):
        # Progressive SH
        cur_sh = min(step // sh_interval, sh_degree)

        # Get training sample
        data = trainset[step % len(trainset)]
        gt_pixels = data["image"].float().to(device) / 255.0  # [H, W, 3]
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        height, width = gt_pixels.shape[:2]

        # Load depth prior if available
        depth_gt = None
        if depth_enabled and depth_dir is not None:
            image_name = parser.image_names[trainset.indices[step % len(trainset)]]
            depth_gt = load_depth_map(depth_dir, image_name)
            if depth_gt is not None:
                depth_gt = depth_gt.to(device)

        # Forward
        color, depth, alphas, meta = render_step(
            splats, camtoworld, K, width, height, cur_sh, config,
        )

        # Loss
        loss, loss_dict = compute_loss(
            color, gt_pixels[None], depth, depth_gt, config, scene_scale,
        )

        # Pre-backward (strategy)
        strategy.step_pre_backward(
            params=splats, optimizers=optimizers,
            state=strategy_state, step=step, info=meta,
        )

        # Backward
        loss.backward()

        # Optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers.values():
            sched.step()

        # Post-backward (densification)
        if strategy_type == "mcmc":
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=meta,
                lr=schedulers["means"].get_last_lr()[0],
            )
        else:
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=meta, packed=True,
            )

        # Logging
        if step % 100 == 0:
            psnr = -10 * math.log10(max(loss_dict["l1"] ** 2, 1e-10)) if loss_dict["l1"] > 0 else 0
            n_gaussians = splats["means"].shape[0]
            elapsed = time.time() - start_time
            msg = (
                f"Step {step:5d}/{max_steps} | "
                f"Loss: {loss_dict['total']:.4f} | "
                f"L1: {loss_dict['l1']:.4f} | "
                f"SSIM: {loss_dict['ssim']:.4f} | "
                f"#GS: {n_gaussians:,} | "
                f"Time: {elapsed:.0f}s"
            )
            if "depth" in loss_dict:
                msg += f" | Depth: {loss_dict['depth']:.4f}"
            logger.info(msg)

            metrics_log.append({
                "step": step, **loss_dict,
                "n_gaussians": n_gaussians,
                "elapsed": elapsed,
            })

        # Evaluation
        if (step + 1) in eval_steps:
            eval_psnr = evaluate(splats, valset, config, device)
            logger.info(f"  [EVAL step {step+1}] Val PSNR: {eval_psnr:.2f} dB")
            if eval_psnr > best_psnr:
                best_psnr = eval_psnr

        # Save checkpoint
        if (step + 1) in save_steps:
            ckpt_path = run_dir / f"step_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "splats": {k: v.data for k, v in splats.items()},
                "config": config,
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    # ── Final save ───────────────────────────────────────────────
    total_time = time.time() - start_time
    final_path = run_dir / "final.pt"
    torch.save({
        "step": max_steps,
        "splats": {k: v.data for k, v in splats.items()},
        "config": config,
    }, final_path)

    # Save metrics
    metrics_path = run_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Training complete: {run_label}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"  Final Gaussians: {splats['means'].shape[0]:,}")
    logger.info(f"  Best val PSNR: {best_psnr:.2f} dB")
    logger.info(f"  Checkpoint: {final_path}")
    logger.info("=" * 60)

    return {
        "run_label": run_label,
        "run_dir": str(run_dir),
        "total_time": total_time,
        "final_gaussians": splats["means"].shape[0],
        "best_psnr": best_psnr,
        "metrics_log": metrics_log,
    }


@torch.no_grad()
def evaluate(splats, valset, config, device):
    """Evaluate PSNR on validation set."""
    total_psnr = 0.0
    sh_degree = config["gaussians"]["sh_degree"]

    for i in range(len(valset)):
        data = valset[i]
        gt_pixels = data["image"].float().to(device) / 255.0
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        height, width = gt_pixels.shape[:2]

        color, _, _, _ = render_step(
            splats, camtoworld, K, width, height, sh_degree, config,
        )

        mse = F.mse_loss(color.squeeze(0), gt_pixels)
        psnr = -10 * math.log10(max(mse.item(), 1e-10))
        total_psnr += psnr

    return total_psnr / max(len(valset), 1)


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode", choices=["baseline", "depth", "mcmc", "all"], default="all",
        help="Training mode: baseline, depth-regularized, mcmc, or all",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps

    all_results = {}

    if args.mode in ("baseline", "all"):
        result = train(config.copy(), "baseline", depth_enabled=False, strategy_type="default")
        all_results["baseline"] = result
        torch.cuda.empty_cache()

    if args.mode in ("depth", "all"):
        result = train(config.copy(), "depth_reg", depth_enabled=True, strategy_type="default")
        all_results["depth_reg"] = result
        torch.cuda.empty_cache()

    if args.mode in ("mcmc", "all"):
        result = train(config.copy(), "mcmc", depth_enabled=False, strategy_type="mcmc")
        all_results["mcmc"] = result
        torch.cuda.empty_cache()

    # Save summary
    results_dir = Path(config["paths"]["results_dir"])
    summary = {}
    for label, res in all_results.items():
        summary[label] = {
            "best_psnr": res["best_psnr"],
            "total_time": res["total_time"],
            "final_gaussians": res["final_gaussians"],
            "run_dir": res["run_dir"],
        }

    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for label, s in summary.items():
        logger.info(f"  {label}: PSNR={s['best_psnr']:.2f} dB, "
                     f"Time={s['total_time']:.0f}s, "
                     f"#GS={s['final_gaussians']:,}")


if __name__ == "__main__":
    main()
