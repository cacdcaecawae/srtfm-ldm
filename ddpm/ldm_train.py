import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import get_h5_dataloader
from ddpm.ddpm_simple import DDPM
from logger import Logger
from network.network import (build_network, convnet_big_cfg, convnet_medium_cfg,
                             convnet_small_cfg, unet_1_cfg, unet_res_cfg,
                             unet_res_diffusion_cfg)


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "convnet_small": convnet_small_cfg,
    "convnet_medium": convnet_medium_cfg,
    "convnet_big": convnet_big_cfg,
    "unet": unet_1_cfg,
    "unet_res": unet_res_cfg,
    "unet_res_diffusion": unet_res_diffusion_cfg,
}

DEFAULT_CONFIG_PATH = Path(__file__).with_name("ddpm_baseline.json")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_replace(src: Path, dst: Path, retries: int = 5, delay: float = 0.1) -> None:
    """Work around Windows file locking when replacing checkpoints."""
    last_error: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_error = exc
            if dst.exists():
                try:
                    dst.unlink()
                except PermissionError:
                    time.sleep(delay)
            time.sleep(delay)
    raise last_error if last_error else PermissionError(f"Failed to replace {dst}")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None:
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def load_vae(model_cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, float]:
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as exc:
        raise ImportError("diffusers is required when use_vae is enabled.") from exc

    vae_path = model_cfg.get("vae_path")
    if not vae_path:
        raise ValueError("vae_path is required when use_vae is enabled.")

    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device)
    vae.half()
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    scale = float(model_cfg.get("vae_scale", getattr(vae.config, "scaling_factor", 1.0)))
    return vae, scale


def _maybe_repeat_channels(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


@torch.no_grad()
def encode_latent(vae: nn.Module, images: torch.Tensor, scale: float, sample: bool) -> torch.Tensor:
    images = _maybe_repeat_channels(images)
    with torch.amp.autocast('cuda'):
        posterior = vae.encode(images).latent_dist
        latents = posterior.sample() if sample else posterior.mode()
    return latents * scale
@torch.no_grad()
def decode_latent(vae: nn.Module, latents: torch.Tensor, scale: float) -> torch.Tensor:
    latents = latents / scale
    with torch.amp.autocast('cuda'):
        decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    if decoded.shape[1] > 1:
        decoded = decoded[:, :1]
    return decoded


@torch.no_grad()
def build_condition_latent(
    vae: nn.Module,
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
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_sd = ema_model.state_dict()
    model_sd = model.state_dict()
    for key, value in model_sd.items():
        ema_sd[key].mul_(decay).add_(value, alpha=1 - decay)
    ema_model.load_state_dict(ema_sd)


def warmup_cosine(optimizer: torch.optim.Optimizer,
                  current_epoch: int,
                  max_epoch: int,
                  lr_min: float = 0.0,
                  lr_max: float = 1e-4,
                  warmup_epoch: int = 10) -> None:
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(
            math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def to_jet(x: torch.Tensor,
           vmin: Optional[float] = None,
           vmax: Optional[float] = None,
           bins: int = 256) -> torch.Tensor:
    """Map a batch of 1-channel tensors to RGB using a jet colormap."""
    squeeze_back = False
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        squeeze_back = True
    elif x.dim() == 3:
        x = x.unsqueeze(0)

    device = x.device
    if vmin is None:
        vmin = float(x.min().item())
    if vmax is None:
        vmax = float(x.max().item())
    if vmax == vmin:
        vmax = vmin + 1e-6

    x_norm = (x - vmin) / (vmax - vmin)
    x_norm = x_norm.clamp(0, 1)

    try:
        cmap = cm.get_cmap("jet")
        lut_np = cmap(np.linspace(0, 1, bins))[:, :3]
    except AttributeError:
        lut_np = cm.get_cmap("jet", bins)(np.linspace(0, 1, bins))[:, :3]

    lut = torch.from_numpy(lut_np).to(device=device, dtype=torch.float32)
    idx = (x_norm * (bins - 1)).round().long()
    idx = idx.squeeze(1)
    rgb = lut[idx]
    rgb = rgb.permute(0, 3, 1, 2).contiguous()

    if squeeze_back:
        rgb = rgb.squeeze(0)
    return rgb


def make_preview_grid(tensor: torch.Tensor, channels: int,
                      nrow: int) -> torch.Tensor:
    if channels == 1:
        tensor = to_jet(tensor, vmin=0.0, vmax=1.0)
    elif channels == 3:
        tensor = to_jet(tensor[:, 0:1], vmin=0.0, vmax=1.0)
    return make_grid(tensor, nrow=nrow)


def build_ddpm(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    if not model_cfg.get("use_vae", False):
        raise ValueError("use_vae must be true for LDM training.")

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
        raise KeyError(f"Unknown DDPM backbone '{ddpm_backbone_key}'")
    ddpm_cfg = MODEL_CONFIGS[ddpm_backbone_key].copy()
    n_steps = model_cfg["diffusion_steps"]

    ddpm_net = build_network(
        ddpm_cfg,
        in_channels=latent_channels,
        image_size=latent_size,
        lr_channels=lr_channels,
        n_steps=n_steps,
    ).to(device)
    return ddpm_net
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


def maybe_load_checkpoint(ddpm_net: nn.Module,
                          cfg: Dict[str, Any],
                          device: torch.device) -> None:
    model_cfg = cfg["model"]
    ddpm_ckpt_path = model_cfg.get("ddpm_checkpoint")
    if ddpm_ckpt_path:
        print(f"Loading DDPM from {ddpm_ckpt_path}")
        ddpm_checkpoint = torch.load(ddpm_ckpt_path, map_location=device)
        ddpm_state_dict = select_state_dict(ddpm_checkpoint)
        ddpm_net.load_state_dict(ddpm_state_dict)
        print("DDPM checkpoint loaded")


def get_image_shape_from_config(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
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


def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader:
    data_cfg = cfg["data"]

    coord_range = None
    if data_cfg.get("use_tfm_channels", False):
        coord_range_x = tuple(data_cfg["coord_range_x"]) if "coord_range_x" in data_cfg else (-1.0, 1.0)
        coord_range_y = tuple(data_cfg["coord_range_y"]) if "coord_range_y" in data_cfg else (-1.0, 1.0)
        coord_range = (coord_range_x, coord_range_y)

    return get_h5_dataloader(
        h5_path=data_cfg["h5_path"],
        batch_size=data_cfg["batch_size"],
        lr_key=data_cfg.get("h5_lr_key", "TFM"),
        hr_key=data_cfg.get("h5_hr_key", "hr"),
        lr_dataset_name=data_cfg.get("h5_lr_dataset"),
        hr_dataset_name=data_cfg.get("h5_hr_dataset"),
        transpose_lr=data_cfg.get("transpose_lr", False),
        transpose_hr=data_cfg.get("transpose_hr", False),
        use_tfm_channels=data_cfg.get("use_tfm_channels", False),
        coord_range=coord_range,
        augment=data_cfg.get("augment", False),
        h_flip_prob=data_cfg.get("h_flip_prob", 0.5),
        translate_prob=data_cfg.get("translate_prob", 0.5),
        max_translate_ratio=data_cfg.get("max_translate_ratio", 0.05),
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=True,
    )


def create_train_val_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    data_cfg = cfg["data"]
    base_loader = create_dataloader(cfg)
    dataset = base_loader.dataset
    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    if val_ratio <= 0.0 or len(dataset) < 2:
        return base_loader, None

    n_val = int(len(dataset) * val_ratio)
    n_val = max(1, min(n_val, len(dataset) - 1))
    n_train = len(dataset) - n_val
    seed = cfg.get("seed")
    generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    num_workers = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train(ddpm: DDPM,
          ddpm_net: nn.Module,
          vae: nn.Module,
          vae_scale: float,
          cfg: Dict[str, Any],
          device: torch.device,
          ckpt_path: Path,
          log_dir: Path,
          log: Logger) -> nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimization"]
    logging_cfg = cfg["logging"]

    use_tfm_channels = data_cfg.get("use_tfm_channels", False)
    latent_channels = int(model_cfg.get("latent_channels", 4))
    latent_downscale = int(model_cfg.get("latent_downscale", 8))

    log.info("Training single-stage latent diffusion (LDM).")
    log.info(f"VAE scale: {vae_scale:.6f}")

    writer = SummaryWriter(log_dir=str(log_dir))

    train_loader, val_loader = create_train_val_loaders(cfg)
    total_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset) if val_loader is not None else 0

    sample_lr, sample_hr, _ = next(iter(train_loader))
    actual_image_size = sample_hr.shape[-1]
    if actual_image_size % latent_downscale != 0:
        raise ValueError(
            f"image_size {actual_image_size} must be divisible by latent_downscale {latent_downscale}."
        )
    latent_size = actual_image_size // latent_downscale

    log.info(
        f"Start training, batch size: {data_cfg['batch_size']}, epochs: {opt_cfg['epochs']}"
    )
    if val_loader is None:
        log.info(
            "Total samples: %d, steps per epoch: %d, image size: %dx%d, "
            "HR channels: %d, LR channels: %d, latent: %dx%dx%d",
            total_samples,
            len(train_loader),
            actual_image_size,
            actual_image_size,
            sample_hr.shape[1],
            sample_lr.shape[1],
            latent_channels,
            latent_size,
            latent_size,
        )
    else:
        log.info(
            "Total samples: %d, val samples: %d, steps per epoch: %d, image size: %dx%d, "
            "HR channels: %d, LR channels: %d, latent: %dx%dx%d",
            total_samples,
            val_samples,
            len(train_loader),
            actual_image_size,
            actual_image_size,
            sample_hr.shape[1],
            sample_lr.shape[1],
            latent_channels,
            latent_size,
            latent_size,
        )

    use_multi_gpu = cfg.get("multi_gpu", False)
    if use_multi_gpu and torch.cuda.device_count() > 1:
        log.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        ddpm_net = nn.DataParallel(ddpm_net)

    ddpm_net = ddpm_net.to(device).train()
    ema_net = deepcopy(ddpm_net).eval().requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        ddpm_net.parameters(),
        lr=opt_cfg["learning_rate"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.99))),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    best_loss = float("inf")
    best_loss_epoch = -1
    model_best_state_dict = None
    ema_model_best_state_dict = None

    use_amp = bool(opt_cfg.get("amp", device.type == "cuda"))
    amp_dtype_str = opt_cfg.get("amp_dtype", "float16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(
        enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16
    )
    log.info(f"Using AMP: {use_amp}, dtype: {amp_dtype_str if use_amp else 'float32'}")
    clip_grad_norm = opt_cfg.get("clip_grad_norm")
    if clip_grad_norm is not None:
        log.info(f"Gradient clipping enabled: {clip_grad_norm}")

    epochs = opt_cfg["epochs"]
    warmup_epochs = opt_cfg.get("warmup_epochs", max(1, epochs // 10))
    lr_min = opt_cfg.get("lr_min", optimizer.param_groups[0]["lr"])
    lr_max = opt_cfg.get("lr_max", optimizer.param_groups[0]["lr"])
    preview_interval = logging_cfg.get("sample_interval", 10)
    preview_count = max(1, logging_cfg.get("num_preview_samples", 4))
    preview_nrow = int(preview_count**0.5)
    preview_nrow = max(1, preview_nrow)

    for epoch in range(epochs):
        total_loss = 0.0
        warmup_cosine(
            optimizer,
            epoch,
            epochs - 1,
            lr_min=lr_min,
            lr_max=lr_max,
            warmup_epoch=warmup_epochs,
        )

        with log.progress_bar(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for lr_images, hr_images, _ in pbar:
                lr_images = lr_images.to(device, non_blocking=True)
                hr_images = hr_images.to(device, non_blocking=True)
                batch_size = hr_images.size(0)

                hr_latent = encode_latent(vae, hr_images, vae_scale, sample=True)
                condition = build_condition_latent(vae, lr_images, vae_scale, use_tfm_channels)

                t = torch.randint(
                    0, ddpm.n_steps, size=(batch_size,), device=device
                )
                eps = torch.randn_like(hr_latent)
                x_t = ddpm.sample_forward(hr_latent, t, eps)

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp and device.type == "cuda",
                ):
                    eps_pred = ddpm_net(x_t, t, condition)
                    loss = loss_fn(eps_pred, eps)

                optimizer.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddpm_net.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(ddpm_net.parameters(), clip_grad_norm)
                    optimizer.step()

                total_loss += loss.item() * batch_size
                ema_update(ema_net, ddpm_net, decay=opt_cfg["ema_decay"])
                pbar.update_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = None
        if val_loader is not None:
            net_was_training = ddpm_net.training
            ddpm_net.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for val_lr_images, val_hr_images, _ in val_loader:
                    val_lr_images = val_lr_images.to(device, non_blocking=True)
                    val_hr_images = val_hr_images.to(device, non_blocking=True)
                    batch_size = val_hr_images.size(0)

                    hr_latent = encode_latent(vae, val_hr_images, vae_scale, sample=True)
                    condition = build_condition_latent(vae, val_lr_images, vae_scale, use_tfm_channels)

                    t = torch.randint(
                        0, ddpm.n_steps, size=(batch_size,), device=device
                    )
                    eps = torch.randn_like(hr_latent)
                    x_t = ddpm.sample_forward(hr_latent, t, eps)

                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp and device.type == "cuda",
                    ):
                        eps_pred = ddpm_net(x_t, t, condition)
                        loss = loss_fn(eps_pred, eps)

                    val_loss += loss.item() * batch_size
            avg_val_loss = val_loss / len(val_loader.dataset)
            writer.add_scalar("val/loss", avg_val_loss, epoch + 1)
            if net_was_training:
                ddpm_net.train()

        if epoch % preview_interval == 0:
            net_was_training = ddpm_net.training
            ddpm_net.eval()
            ema_net.eval()
            with torch.inference_mode():
                preview_batch = min(preview_count, lr_images.size(0))
                lr_subset = lr_images[:preview_batch]
                hr_subset = hr_images[:preview_batch]

                condition_subset = build_condition_latent(
                    vae, lr_subset, vae_scale, use_tfm_channels
                )
                img_shape = get_image_shape_from_config(cfg)
                img_latent = ddpm.sample_backward_sr(
                    (preview_batch, *img_shape),
                    ddpm_net,
                    condition_subset,
                    device=device,
                    simple_var=True,
                )
                img_ema_latent = ddpm.sample_backward_sr(
                    (preview_batch, *img_shape),
                    ema_net,
                    condition_subset,
                    device=device,
                    simple_var=True,
                )
                img_net = decode_latent(vae, img_latent, vae_scale).cpu()
                img_ema = decode_latent(vae, img_ema_latent, vae_scale).cpu()
            if net_was_training:
                ddpm_net.train()

            lr01 = ((lr_subset.detach().cpu().clamp(-1, 1) + 1) / 2)
            hr01 = ((hr_subset.detach().cpu().clamp(-1, 1) + 1) / 2)
            net01 = ((img_net.detach().cpu().clamp(-1, 1) + 1) / 2)
            ema01 = ((img_ema.detach().cpu().clamp(-1, 1) + 1) / 2)
            channels = lr01.shape[1]
            writer.add_image(
                f"sample/epoch_{epoch + 1}_lr",
                make_preview_grid(lr01, channels, preview_nrow),
                epoch + 1,
            )
            writer.add_image(
                f"sample/epoch_{epoch + 1}_hr",
                make_preview_grid(hr01, channels, preview_nrow),
                epoch + 1,
            )
            writer.add_image(
                f"sample/epoch_{epoch + 1}_net",
                make_preview_grid(net01, channels, preview_nrow),
                epoch + 1,
            )
            writer.add_image(
                f"sample/epoch_{epoch + 1}_ema",
                make_preview_grid(ema01, channels, preview_nrow),
                epoch + 1,
            )

        avg_loss = total_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/learning_rate", current_lr, epoch + 1)

        if avg_val_loss is None:
            log.info(
                f"Epoch {epoch + 1}/{epochs} finished. "
                f"Average loss: {avg_loss:.6f}. "
                f"LR: {current_lr:.2e}"
            )
        else:
            log.info(
                f"Epoch {epoch + 1}/{epochs} finished. "
                f"Train loss: {avg_loss:.6f}. "
                f"Val loss: {avg_val_loss:.6f}. "
                f"LR: {current_lr:.2e}"
            )

        metric_loss = avg_val_loss if avg_val_loss is not None else avg_loss
        if metric_loss < best_loss:
            best_loss = metric_loss
            best_loss_epoch = epoch + 1
            model_state = (
                ddpm_net.module.state_dict()
                if isinstance(ddpm_net, nn.DataParallel)
                else ddpm_net.state_dict()
            )
            ema_state = (
                ema_net.module.state_dict()
                if isinstance(ema_net, nn.DataParallel)
                else ema_net.state_dict()
            )
            model_best_state_dict = deepcopy(model_state)
            ema_model_best_state_dict = deepcopy(ema_state)

            ckpt_best = {
                "epoch": best_loss_epoch,
                "ema_decay": opt_cfg["ema_decay"],
                "best_loss": best_loss,
                "model_best_state_dict": model_best_state_dict,
                "ema_model_best_state_dict": ema_model_best_state_dict,
            }
            best_path = ckpt_path.with_name(f"{ckpt_path.stem}_best.pth")
            tmp_best = best_path.with_suffix(best_path.suffix + ".tmp")
            torch.save(ckpt_best, tmp_best)
            safe_replace(tmp_best, best_path)

        model_state = (
            ddpm_net.module.state_dict()
            if isinstance(ddpm_net, nn.DataParallel)
            else ddpm_net.state_dict()
        )
        ema_state = (
            ema_net.module.state_dict()
            if isinstance(ema_net, nn.DataParallel)
            else ema_net.state_dict()
        )

        ckpt = {
            "epoch": epoch + 1,
            "ema_decay": opt_cfg["ema_decay"],
            "model_state_dict": model_state,
            "ema_model_state_dict": ema_state,
            "best_loss": best_loss,
            "best_loss_epoch": best_loss_epoch,
            "model_best_state_dict": model_best_state_dict,
            "ema_model_best_state_dict": ema_model_best_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(ckpt, tmp_ckpt)
        safe_replace(tmp_ckpt, ckpt_path)

    writer.close()
    log.info("Done training!")
    return ema_net


def main() -> None:
    user_log = input("Optional note (press Enter to skip): ")
    log = Logger(rank=0, log_dir="runs/logs")

    log.info("=======================================================")
    log.info("        LDM Super-Resolution Trainer")
    log.info("=======================================================")

    if user_log.strip():
        log.info(f"User Note: {user_log.strip()}")
        log.info("-------------------------------------------------------")

    cfg = load_config(DEFAULT_CONFIG_PATH)
    seed = cfg.get("seed", 42)
    if seed is not None:
        set_seed(seed)
        log.info(f"Random seed: {seed}")
    else:
        log.warning("Seed disabled; results will vary between runs.")

    device = resolve_device(cfg.get("device"))
    log.info(f"Using device: {device}")

    model_cfg = cfg["model"]
    if not model_cfg.get("use_vae", False):
        raise ValueError("use_vae must be true for LDM training.")

    if cfg["data"].get("augment", False):
        log.info("Data augmentation enabled.")
    if cfg["data"].get("use_tfm_channels", False):
        log.info("Using 3-channel TFM as conditioning input.")
    else:
        log.info("Using single-channel LR as conditioning input.")

    vae, vae_scale = load_vae(model_cfg, device)
    ddpm_net = build_ddpm(cfg, device)
    maybe_load_checkpoint(ddpm_net, cfg, device)

    n_steps = cfg["model"]["diffusion_steps"]
    ddpm = DDPM(device, n_steps)

    ckpt_dir = Path(cfg["model"]["checkpoint_dir"])
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / cfg["model"]["checkpoint_name"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_cfg = cfg["logging"]
    log_root = Path(log_cfg["tensorboard_root"])
    ensure_dir(log_root)
    log_dir = log_root / f"{timestamp}-{log_cfg.get('experiment_name', 'ldm-train')}"
    ensure_dir(log_dir)

    train(ddpm, ddpm_net, vae, vae_scale, cfg, device, ckpt_path, log_dir, log)

    cfg_copy_path = log_dir / "train_config.json"
    with cfg_copy_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)
    log.info(f"Configuration saved to {cfg_copy_path}")


if __name__ == "__main__":
    main()


