import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image
from torchmetrics.functional import (peak_signal_noise_ratio,
                                     structural_similarity_index_measure)
from tqdm import tqdm

from dataset import get_h5_dataloader
from ddpm.ddim import DDIM
from ddpm.ddpm_simple import DDPM
from network.network import (build_network, convnet_big_cfg, convnet_medium_cfg,
                             convnet_small_cfg, unet_1_cfg, unet_res_cfg,
                             unet_res_diffusion_cfg)

JET_PALETTE: List[int] = (
    (cm.jet(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0)
    .astype(np.uint8)
    .reshape(-1)
    .tolist()
)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "convnet_small": convnet_small_cfg,
    "convnet_medium": convnet_medium_cfg,
    "convnet_big": convnet_big_cfg,
    "unet": unet_1_cfg,
    "unet_res": unet_res_cfg,
    "unet_res_diffusion": unet_res_diffusion_cfg,
}

DEFAULT_CONFIG_PATH = Path(__file__).with_name("ddpm_eval.json")
DEFAULT_FALLBACK_CONFIG_PATH = Path(__file__).with_name("ddpm_baseline.json")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None:
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_vae(model_cfg: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, float]:
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as exc:
        raise ImportError("diffusers is required when use_vae is enabled.") from exc

    vae_path = model_cfg.get("vae_path")
    if not vae_path:
        raise ValueError("vae_path is required for LDM evaluation.")

    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    scale = float(model_cfg.get("vae_scale", getattr(vae.config, "scaling_factor", 1.0)))
    return vae, scale


def maybe_fill_model_defaults(cfg: Dict[str, Any], fallback_path: Path) -> Dict[str, Any]:
    if not fallback_path.exists():
        return cfg
    fallback = load_config(fallback_path)
    cfg_model = cfg.setdefault("model", {})
    fallback_model = fallback.get("model", {})
    for key in ("use_vae", "vae_path", "vae_scale", "latent_channels", "latent_downscale"):
        if key not in cfg_model and key in fallback_model:
            cfg_model[key] = fallback_model[key]
    return cfg


def resolve_ddpm_checkpoint(model_cfg: Dict[str, Any]) -> str:
    ddpm_ckpt_path = model_cfg.get("ddpm_checkpoint")
    if ddpm_ckpt_path:
        return ddpm_ckpt_path
    ckpt_dir = model_cfg.get("checkpoint_dir")
    ckpt_name = model_cfg.get("checkpoint_name")
    if ckpt_dir and ckpt_name:
        return str(Path(ckpt_dir) / ckpt_name)
    raise ValueError("ddpm_checkpoint is required for LDM evaluation.")


def normalize_sampler_type(sampler_type: Optional[str]) -> str:
    if sampler_type is None:
        return "ddpm"
    sampler_type = str(sampler_type).strip().lower()
    if sampler_type in ("ddpm", "ddim"):
        return sampler_type
    raise ValueError(f"Unknown sampler type '{sampler_type}'. Use 'ddpm' or 'ddim'.")


def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader:
    data_cfg = cfg["data"]
    coord_range = None
    if data_cfg.get("use_tfm_channels", False):
        coord_range_x = tuple(data_cfg["coord_range_x"]) if "coord_range_x" in data_cfg else (-1.0, 1.0)
        coord_range_y = tuple(data_cfg["coord_range_y"]) if "coord_range_y" in data_cfg else (-1.0, 1.0)
        coord_range = (coord_range_x, coord_range_y)

    return get_h5_dataloader(
        h5_path=data_cfg["h5_path"],
        batch_size=data_cfg.get("batch_size", 1),
        lr_key=data_cfg.get("h5_lr_key", "TFM"),
        hr_key=data_cfg.get("h5_hr_key", "hr"),
        lr_dataset_name=data_cfg.get("h5_lr_dataset"),
        hr_dataset_name=data_cfg.get("h5_hr_dataset"),
        transpose_lr=data_cfg.get("transpose_lr", False),
        transpose_hr=data_cfg.get("transpose_hr", False),
        use_tfm_channels=data_cfg.get("use_tfm_channels", False),
        coord_range=coord_range,
        augment=False,
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=False,
    )


def select_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        "ema_model_best_state_dict",
        "model_best_state_dict",
        "ema_model_state_dict",
        "model_state_dict",
    ):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def build_ddpm_net(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    if not model_cfg.get("use_vae", False):
        raise ValueError("use_vae must be true for LDM evaluation.")

    latent_channels = int(model_cfg.get("latent_channels", 4))
    latent_downscale = int(model_cfg.get("latent_downscale", 8))
    image_size = data_cfg["image_size"]
    if image_size % latent_downscale != 0:
        raise ValueError(
            f"image_size {image_size} must be divisible by latent_downscale {latent_downscale}."
        )
    latent_size = image_size // latent_downscale

    use_tfm_channels = data_cfg.get("use_tfm_channels", False)
    if use_tfm_channels:
        lr_channels = latent_channels + 2
    else:
        lr_channels = latent_channels

    ddpm_backbone_key = model_cfg.get("ddpm_backbone", "unet_res_diffusion")
    if ddpm_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown DDPM backbone '{ddpm_backbone_key}'.")
    ddpm_cfg = MODEL_CONFIGS[ddpm_backbone_key].copy()

    n_steps = int(model_cfg["diffusion_steps"])
    ddpm_net = build_network(
        ddpm_cfg,
        in_channels=latent_channels,
        image_size=latent_size,
        lr_channels=lr_channels,
        n_steps=n_steps,
    ).to(device)

    ddpm_ckpt_path = resolve_ddpm_checkpoint(model_cfg)
    ddpm_checkpoint = torch.load(ddpm_ckpt_path, map_location=device)
    ddpm_state_dict = select_state_dict(ddpm_checkpoint)
    ddpm_net.load_state_dict(ddpm_state_dict)
    ddpm_net.eval()

    return ddpm_net


def get_latent_shape(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    latent_channels = int(model_cfg.get("latent_channels", 4))
    latent_downscale = int(model_cfg.get("latent_downscale", 8))
    image_size = data_cfg["image_size"]
    if image_size % latent_downscale != 0:
        raise ValueError(
            f"image_size {image_size} must be divisible by latent_downscale {latent_downscale}."
        )
    latent_size = image_size // latent_downscale
    return (latent_channels, latent_size, latent_size)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp(-1, 1).add(1).div(2)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


def tensor_to_image(tensor: torch.Tensor, apply_jet: bool = False) -> Image.Image:
    array = tensor.permute(1, 2, 0).cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    if apply_jet:
        if array.shape[2] == 1:
            gray_np = array[:, :, 0]
        else:
            gray_np = array.mean(axis=2)
        gray_uint8 = (gray_np * 255.0).astype(np.uint8)
        jet_image = Image.fromarray(gray_uint8, mode="P")
        jet_image.putpalette(JET_PALETTE)
        return jet_image
    array = (array * 255.0).astype(np.uint8)
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0], mode="L")
    return Image.fromarray(array)


def _maybe_repeat_channels(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


@torch.no_grad()
def encode_latent(vae: torch.nn.Module, images: torch.Tensor, scale: float, sample: bool) -> torch.Tensor:
    images = _maybe_repeat_channels(images)
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample() if sample else posterior.mode()
    return latents * scale


@torch.no_grad()
def decode_latent(vae: torch.nn.Module, latents: torch.Tensor, scale: float) -> torch.Tensor:
    latents = latents / scale
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    if decoded.shape[1] > 1:
        decoded = decoded[:, :1]
    return decoded


@torch.no_grad()
def build_condition_latent(
    vae: torch.nn.Module,
    lr_images: torch.Tensor,
    scale: float,
    use_tfm_channels: bool,
) -> torch.Tensor:
    if not use_tfm_channels:
        return encode_latent(vae, lr_images, scale, sample=False)
    if lr_images.shape[1] < 3:
        raise ValueError("TFM channels expected, but lr_images has fewer than 3 channels.")

    i_latent = encode_latent(vae, lr_images[:, :1], scale, sample=False)
    xy = lr_images[:, 1:3]
    xy_down = F.interpolate(
        xy,
        size=i_latent.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    return torch.cat([i_latent, xy_down], dim=1)


def evaluate(cfg: Dict[str, Any], device: torch.device) -> None:
    data_cfg = cfg["data"]
    sampler_cfg = cfg.get("sampler", {})
    sampler_type = normalize_sampler_type(sampler_cfg.get("type"))

    dataloader = create_dataloader(cfg)
    vae, vae_scale = load_vae(cfg["model"], device)
    ddpm_net = build_ddpm_net(cfg, device)

    use_jet = data_cfg.get("channels", 1) == 1
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "SR/eval_results_ldm"))
    images_dir = output_root / "images"
    ensure_dir(images_dir)
    ensure_dir(output_root)

    threshold = float(sampler_cfg.get("threshold", 0.05))
    seed = sampler_cfg.get("seed", 1234)
    simple_var = bool(sampler_cfg.get("simple_var", True))
    ddim_steps = int(sampler_cfg.get("ddim_steps", 20))
    eta = float(sampler_cfg.get("eta", 1.0))
    if sampler_type == "ddim" and ddim_steps <= 0:
        raise ValueError("ddim_steps must be a positive integer.")

    model_cfg = cfg["model"]
    n_steps = int(model_cfg["diffusion_steps"])
    min_beta = float(model_cfg.get("min_beta", 1e-4))
    max_beta = float(model_cfg.get("max_beta", 2e-2))
    if sampler_type == "ddim":
        ddpm = DDIM(device, n_steps, min_beta=min_beta, max_beta=max_beta)
    else:
        ddpm = DDPM(device, n_steps, min_beta=min_beta, max_beta=max_beta)

    latent_shape = get_latent_shape(cfg)

    psnr_scores = []
    ssim_scores = []
    iou_scores = []
    per_image_results = []

    with torch.inference_mode():
        for lr_images, hr_images, names in tqdm(dataloader, desc="Evaluating LDM"):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            condition = build_condition_latent(
                vae, lr_images, vae_scale, data_cfg.get("use_tfm_channels", False)
            )

            batch_size = hr_images.size(0)
            img_shape = (batch_size, *latent_shape)
            if sampler_type == "ddim":
                latent_images = ddpm.sample_backward_sr(
                    img_shape,
                    ddpm_net,
                    condition,
                    device=device,
                    simple_var=simple_var,
                    ddim_step=ddim_steps,
                    eta=eta,
                )
            else:
                latent_images = ddpm.sample_backward_sr(
                    img_shape,
                    ddpm_net,
                    condition,
                    device=device,
                    simple_var=simple_var,
                )

            sr_images = decode_latent(vae, latent_images, vae_scale)
            sr_for_metric = denormalize(sr_images)
            sr_for_metric = torch.where(
                sr_for_metric < threshold,
                torch.zeros_like(sr_for_metric),
                sr_for_metric,
            )
            sr_for_image = sr_for_metric
            hr_for_metric = denormalize(hr_images)

            for idx, name in enumerate(names):
                pred = sr_for_metric[idx].unsqueeze(0)
                target = hr_for_metric[idx].unsqueeze(0)

                psnr = peak_signal_noise_ratio(pred, target, data_range=1.0)
                ssim = structural_similarity_index_measure(pred, target, data_range=1.0)
                iou = compute_iou(pred, target, threshold=threshold)

                psnr_scores.append(psnr.item())
                ssim_scores.append(ssim.item())
                iou_scores.append(iou)
                per_image_results.append({
                    "filename": name,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "iou": iou,
                })

                if output_cfg.get("save_images", True):
                    if not any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]):
                        image_filename = f"{name}.png"
                    else:
                        image_filename = name
                    tensor_to_image(sr_for_image[idx], apply_jet=use_jet).save(images_dir / image_filename)

    avg_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    max_psnr = float(np.max(psnr_scores)) if psnr_scores else 0.0
    min_psnr = float(np.min(psnr_scores)) if psnr_scores else 0.0

    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    max_ssim = float(np.max(ssim_scores)) if ssim_scores else 0.0
    min_ssim = float(np.min(ssim_scores)) if ssim_scores else 0.0

    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    max_iou = float(np.max(iou_scores)) if iou_scores else 0.0
    min_iou = float(np.min(iou_scores)) if iou_scores else 0.0

    print(f"PSNR - Average: {avg_psnr:.4f}, Max: {max_psnr:.4f}, Min: {min_psnr:.4f}")
    print(f"SSIM - Average: {avg_ssim:.4f}, Max: {max_ssim:.4f}, Min: {min_ssim:.4f}")
    print(f"IoU  - Average: {avg_iou:.4f}, Max: {max_iou:.4f}, Min: {min_iou:.4f}")

    results_txt = output_root / "results.txt"
    data_source = cfg["data"].get("h5_path", "N/A")
    with results_txt.open("w", encoding="utf-8") as handle:
        handle.write("LDM Evaluation Summary\n")
        handle.write("======================\n")
        handle.write(f"DDPM: {resolve_ddpm_checkpoint(cfg['model'])}\n")
        handle.write(f"Dataset: {data_source}\n")
        handle.write(f"Sampler: {sampler_type}\n")
        handle.write(f"Threshold: {threshold}\n")
        handle.write(f"Simple var: {simple_var}\n")
        if sampler_type == "ddim":
            handle.write(f"DDIM steps: {ddim_steps}\n")
            handle.write(f"Eta: {eta}\n")
        handle.write("\n")
        handle.write(f"PSNR - Average: {avg_psnr:.4f}, Max: {max_psnr:.4f}, Min: {min_psnr:.4f}\n")
        handle.write(f"SSIM - Average: {avg_ssim:.4f}, Max: {max_ssim:.4f}, Min: {min_ssim:.4f}\n")
        handle.write(f"IoU  - Average: {avg_iou:.4f}, Max: {max_iou:.4f}, Min: {min_iou:.4f}\n")

    results_csv = output_root / "results_per_image.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "psnr", "ssim", "iou"])
        writer.writeheader()
        writer.writerows(per_image_results)

    print(f"Results saved to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LDM SR model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to LDM evaluation configuration JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = maybe_fill_model_defaults(cfg, DEFAULT_FALLBACK_CONFIG_PATH)
    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")
    evaluate(cfg, device)


if __name__ == "__main__":
    main()
