import argparse
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
from matplotlib import cm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import get_h5_dataloader
from logger import Logger
from ddpm.ddpm_simple import DDPM
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

DEFAULT_CONFIG_PATH = Path("configs/ddpm_baseline.json")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
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
        lr = lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
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

    # 修复 matplotlib 弃用警告
    try:
        # matplotlib >= 3.7
        cmap = cm.get_cmap('jet')
        lut_np = cmap(np.linspace(0, 1, bins))[:, :3]
    except AttributeError:
        # matplotlib < 3.7
        lut_np = cm.get_cmap('jet', bins)(np.linspace(0, 1, bins))[:, :3]
    
    lut = torch.from_numpy(lut_np).to(device=device, dtype=torch.float32)

    # 处理单通道: [B, 1, H, W]
    idx = (x_norm * (bins - 1)).round().long()  # [B, 1, H, W]
    idx = idx.squeeze(1)  # [B, H, W]
    rgb = lut[idx]  # [B, H, W, 3]
    rgb = rgb.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

    if squeeze_back:
        rgb = rgb.squeeze(0)
    return rgb


def make_preview_grid(tensor: torch.Tensor, channels: int,
                      nrow: int) -> torch.Tensor:
    """
    创建预览网格图像
    
    Args:
        tensor: [B, C, H, W] - C=1 单通道用jet着色, C=3 直接使用
        channels: 通道数
        nrow: 网格行数
    """
    if channels == 1:
        # 单通道: 转为 RGB jet colormap
        tensor = to_jet(tensor, vmin=0.0, vmax=1.0)
    elif channels == 3:
        # 三通道: 只显示第一个通道 (I通道)
        tensor = to_jet(tensor[:, 0:1], vmin=0.0, vmax=1.0)
    return make_grid(tensor, nrow=nrow)


def save_grid_image(tensor: torch.Tensor, output_path: Path) -> None:
    array = (tensor.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    if array.shape[2] == 1:
        image = Image.fromarray(array[:, :, 0], mode='L')
    else:
        image = Image.fromarray(array, mode='RGB')
    image.save(output_path)


def build_models(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """构建 UNet 和 DDPM 模型。
    
    Returns:
        (unet, ddpm_net): UNet 用于生成粗糙 HR，DDPM 用于细化
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    image_size = data_cfg["image_size"]
    in_channels = data_cfg["channels"]  # HR 的通道数 (1)
    
    # 1. 构建 UNet（用于生成粗糙 HR）
    unet_backbone_key = model_cfg.get("unet_backbone", "unet_res")
    if unet_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown UNet backbone '{unet_backbone_key}'")
    unet_cfg = MODEL_CONFIGS[unet_backbone_key].copy()
    
    # UNet 输入是 TFM 3通道，输出是 HR 1通道
    lr_channels = 3 if data_cfg.get("use_tfm_channels", False) else in_channels
    unet = build_network(unet_cfg, in_channels, image_size, lr_channels, n_steps=None).to(device)
    
    # 2. 构建 DDPM（用于细化）
    ddpm_backbone_key = model_cfg.get("ddpm_backbone", "unet_res_diffusion")
    if ddpm_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown DDPM backbone '{ddpm_backbone_key}'")
    ddpm_cfg = MODEL_CONFIGS[ddpm_backbone_key].copy()
    n_steps = model_cfg["diffusion_steps"]
    
    # DDPM 输入是：粗糙HR(1) + TFM(3) = 4通道条件，输出是噪声预测(1)
    ddpm_lr_channels = in_channels + lr_channels  # 1 + 3 = 4
    ddpm_net = build_network(ddpm_cfg, in_channels, image_size, ddpm_lr_channels, n_steps).to(device)
    
    return unet, ddpm_net


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


def maybe_load_checkpoint(unet: nn.Module, ddpm_net: nn.Module, 
                          cfg: Dict[str, Any], device: torch.device) -> None:
    """加载预训练模型权重。"""
    model_cfg = cfg["model"]
    
    # 加载 UNet 权重（必须，用于生成粗糙 HR）
    unet_ckpt_path = model_cfg.get("unet_checkpoint")
    if not unet_ckpt_path:
        raise ValueError("unet_checkpoint is required for two-stage training")
    
    print(f"Loading UNet from {unet_ckpt_path}")
    unet_checkpoint = torch.load(unet_ckpt_path, map_location=device)
    unet_state_dict = select_state_dict(unet_checkpoint)
    unet.load_state_dict(unet_state_dict)
    unet.eval()  # UNet 冻结，不训练
    for param in unet.parameters():
        param.requires_grad = False
    print("UNet loaded and frozen")
    
    # 加载 DDPM 权重（可选，用于继续训练）
    ddpm_ckpt_path = model_cfg.get("ddpm_checkpoint")
    if ddpm_ckpt_path:
        print(f"Loading DDPM from {ddpm_ckpt_path}")
        ddpm_checkpoint = torch.load(ddpm_ckpt_path, map_location=device)
        ddpm_state_dict = select_state_dict(ddpm_checkpoint)
        ddpm_net.load_state_dict(ddpm_state_dict)
        print("DDPM checkpoint loaded")


def get_image_shape_from_config(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """从配置中获取图像形状 (channels, height, width)"""
    data_cfg = cfg["data"]
    channels = data_cfg["channels"]
    image_size = data_cfg["image_size"]
    return (channels, image_size, image_size)


def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader:
    data_cfg = cfg["data"]
    
    # 构建 coord_range 参数
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



def train(ddpm: DDPM,
          unet: nn.Module,
          ddpm_net: nn.Module,
          cfg: Dict[str, Any],
          device: torch.device,
          ckpt_path: Path,
          log_dir: Path,
          log: Logger) -> nn.Module:
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimization"]
    logging_cfg = cfg["logging"]
    log.info("Training two-stage DDPM with frozen UNet backbone.")

    writer = SummaryWriter(log_dir=str(log_dir))
    
    dataloader = create_dataloader(cfg)
    total_samples = len(dataloader.dataset)
    
    # 获取实际图像尺寸
    sample_lr, sample_hr, _ = next(iter(dataloader))
    actual_image_size = sample_hr.shape[-1]
    
    log.info(f"Start training, batch size: {data_cfg['batch_size']}, epochs: {opt_cfg['epochs']}")
    log.info(f"Total samples: {total_samples}, steps per epoch: {len(dataloader)}, image size: {actual_image_size}x{actual_image_size}, HR channels: {sample_hr.shape[1]}, LR channels: {sample_lr.shape[1]}")
    
    # UNet 已经冻结，只训练 DDPM
    unet.eval()
    ddpm_net = ddpm_net.to(device).train()
    ema_net = deepcopy(ddpm_net).eval().requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        ddpm_net.parameters(),
        lr=opt_cfg["learning_rate"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.99))),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    best_loss = float('inf')
    best_loss_epoch = -1
    model_best_state_dict = None
    ema_model_best_state_dict = None

    use_amp = bool(opt_cfg.get("amp", device.type == 'cuda'))
    amp_dtype_str = opt_cfg.get("amp_dtype", "float16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16)
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
    tic = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        warmup_cosine(optimizer,
                      epoch,
                      epochs - 1,
                      lr_min=lr_min,
                      lr_max=lr_max,
                      warmup_epoch=warmup_epochs)

        # 使用 Rich Progress 进度条
        with log.progress_bar(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for lr_images, hr_images, _ in pbar:
                lr_images = lr_images.to(device, non_blocking=True)
                hr_images = hr_images.to(device, non_blocking=True)
                batch_size = hr_images.size(0)

                # 生成粗糙 HR，作为 condition 的一部分
                with torch.no_grad():
                    coarse_hr = unet(lr_images)
                condition = torch.cat([coarse_hr, lr_images], dim=1)

                # DDPM 前向过程：添加噪声
                t = torch.randint(0,
                                  ddpm.n_steps,
                                  size=(batch_size, ),
                                  device=device)
                eps = torch.randn_like(hr_images)
                x_t = ddpm.sample_forward(hr_images, t, eps)

                with torch.amp.autocast(device_type=device.type,
                                        dtype=amp_dtype,
                                        enabled=use_amp
                                        and device.type == "cuda"):
                    eps_pred = ddpm_net(x_t, t, condition)
                    loss = loss_fn(eps_pred, eps)

                optimizer.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddpm_net.parameters(),
                                                       clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(ddpm_net.parameters(),
                                                       clip_grad_norm)
                    optimizer.step()

                total_loss += loss.item() * batch_size
                ema_update(ema_net, ddpm_net, decay=opt_cfg["ema_decay"])
                pbar.update_postfix(loss=f"{loss.item():.4f}")

        if epoch % preview_interval == 0:
            net_was_training = ddpm_net.training
            ddpm_net.eval()
            ema_net.eval()
            with torch.inference_mode():
                preview_batch = min(preview_count, lr_images.size(0))
                lr_subset = lr_images[:preview_batch]
                
                # 生成粗糙 HR
                coarse_hr_subset = unet(lr_subset)
                condition_subset = torch.cat([coarse_hr_subset, lr_subset], dim=1)
                
                img_shape = get_image_shape_from_config(cfg)
                img_net = ddpm.sample_backward_sr((preview_batch, *img_shape),
                                                  ddpm_net,
                                                  condition_subset,
                                                  device=device,
                                                  simple_var=True).cpu()
                img_ema = ddpm.sample_backward_sr((preview_batch, *img_shape),
                                                  ema_net,
                                                  condition_subset,
                                                  device=device,
                                                  simple_var=True).cpu()
            if net_was_training:
                ddpm_net.train()

            hr_subset = hr_images[:preview_batch].cpu()
            coarse_hr_cpu = coarse_hr_subset.cpu()
            lr01 = ((lr_subset.detach().cpu().clamp(-1, 1) + 1) / 2)
            coarse01 = ((coarse_hr_cpu.detach().clamp(-1, 1) + 1) / 2)
            hr01 = ((hr_subset.clamp(-1, 1) + 1) / 2)
            net01 = ((img_net.detach().cpu().clamp(-1, 1) + 1) / 2)
            ema01 = ((img_ema.detach().cpu().clamp(-1, 1) + 1) / 2)
            channels = lr01.shape[1]
            writer.add_image(f'sample/epoch_{epoch + 1}_lr',
                             make_preview_grid(lr01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_coarse',
                             make_preview_grid(coarse01, 1, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_hr',
                             make_preview_grid(hr01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_net',
                             make_preview_grid(net01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_ema',
                             make_preview_grid(ema01, channels, preview_nrow),
                             epoch + 1)

        avg_loss = total_loss / len(dataloader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/loss', avg_loss, epoch + 1)
        writer.add_scalar('train/learning_rate', current_lr, epoch + 1)

        toc = time.time()
        log.info(
            f"Epoch {epoch + 1}/{epochs} finished. "
            f"Average loss: {avg_loss:.6f}. "
            f"LR: {current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_loss_epoch = epoch + 1
            model_best_state_dict = deepcopy(ddpm_net.state_dict())
            ema_model_best_state_dict = deepcopy(ema_net.state_dict())

            ckpt_best = {
                'epoch': best_loss_epoch,
                'ema_decay': opt_cfg["ema_decay"],
                'best_loss': best_loss,
                'model_best_state_dict': model_best_state_dict,
                'ema_model_best_state_dict': ema_model_best_state_dict,
            }
            best_path = ckpt_path.with_name(f"{ckpt_path.stem}_best.pth")
            tmp_best = best_path.with_suffix(best_path.suffix + ".tmp")
            torch.save(ckpt_best, tmp_best)
            safe_replace(tmp_best, best_path)

        ckpt = {
            'epoch': epoch + 1,
            'ema_decay': opt_cfg["ema_decay"],
            'model_state_dict': ddpm_net.state_dict(),
            'ema_model_state_dict': ema_net.state_dict(),
            'best_loss': best_loss,
            'best_loss_epoch': best_loss_epoch,
            'model_best_state_dict': model_best_state_dict,
            'ema_model_best_state_dict': ema_model_best_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(ckpt, tmp_ckpt)
        safe_replace(tmp_ckpt, ckpt_path)

    writer.close()
    log.info("Done training!")
    return ema_net


def main() -> None:
    # 可选择输入一段记录日志
    user_log = input("请输入一段日志记录（或直接回车跳过）: ")
    
    # 初始化 logger
    log = Logger(rank=0, log_dir="runs/logs")
    
    log.info("=======================================================")
    log.info("        Two-Stage DDPM Super-Resolution Trainer")
    log.info("=======================================================")
    
    # 如果有用户输入，写入日志开头
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

    if cfg["data"].get("augment", False):
        log.info("开启数据增强.")
    if cfg["data"].get("use_tfm_channels", False):
        log.info("使用 3 通道作为低分辨率输入.")
    else:
        log.info("使用单通道作为低分辨率输入.")

    # 构建两个模型：UNet 和 DDPM
    unet, ddpm_net = build_models(cfg, device)
    maybe_load_checkpoint(unet, ddpm_net, cfg, device)
    log.info("Initialized two-stage model: frozen UNet + trainable DDPM.")

    n_steps = cfg["model"]["diffusion_steps"]
    ddpm = DDPM(device, n_steps)

    ckpt_dir = Path(cfg["model"]["checkpoint_dir"])
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / cfg["model"]["checkpoint_name"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_cfg = cfg["logging"]
    log_root = Path(log_cfg["tensorboard_root"])
    ensure_dir(log_root)
    log_dir = log_root / f"{timestamp}-{log_cfg.get('experiment_name', 'sr-train')}"
    ensure_dir(log_dir)
    
    train(ddpm, unet, ddpm_net, cfg, device, ckpt_path, log_dir, log)
    
    # 训练结束后保存配置文件（记录实际完成的训练）
    cfg_copy_path = log_dir / "train_config.json"
    with cfg_copy_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)
    log.info(f"Configuration saved to {cfg_copy_path}")


if __name__ == "__main__":
    main()
