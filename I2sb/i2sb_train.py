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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import get_h5_dataloader
from I2sb.diffusion import Diffusion
from logger import Logger
from network.networksb import (build_network, convnet_big_cfg, convnet_medium_cfg,
                     convnet_small_cfg, unet_res_cfg, unet_res_diffusion_cfg)


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "convnet_small": convnet_small_cfg,
    "convnet_medium": convnet_medium_cfg,
    "convnet_big": convnet_big_cfg,
    "unet_res": unet_res_cfg,
    "unet_res_diffusion": unet_res_diffusion_cfg,
}

DEFAULT_CONFIG_PATH = Path(__file__).with_name("i2sb_twostage.json")


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


def env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


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
        lr = lr_max * (current_epoch + 1) / warmup_epoch  # 从 lr_max/warmup_epoch 开始
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

    # 修复 matplotlib 弃用警告 - 使用新 API
    try:
        # matplotlib >= 3.7
        from matplotlib import colormaps
        cmap = colormaps.get_cmap('jet')
        lut_np = cmap(np.linspace(0, 1, bins))[:, :3]
    except (ImportError, AttributeError):
        # matplotlib < 3.7
        cmap = cm.get_cmap('jet')
        lut_np = cmap(np.linspace(0, 1, bins))[:, :3]
    
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


def build_models(cfg: Dict[str, Any],
                 device: torch.device,
                 noise_levels: Optional[torch.Tensor] = None,
                 two_stage: bool = True) -> Tuple[Optional[nn.Module], nn.Module]:
    """构建 UNet 和 I2SB 模型。
    
    Args:
        cfg: 配置字典
        device: 计算设备
        noise_levels: I2SB的时间编码映射（保留以备将来使用DiT等网络）
    
    Returns:
        (unet, i2sb_net): UNet 用于生成粗糙 HR，I2SB 用于构建桥接
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    image_size = data_cfg["image_size"]
    in_channels = data_cfg["channels"]  # HR 的通道数 (1)
    
    # UNet 输入是 TFM 3通道，输出是 HR 1通道
    lr_channels = 3 if data_cfg.get("use_tfm_channels", False) else in_channels
    unet = None
    if two_stage:
        # 1. 构建 UNet（用于生成粗糙 HR）
        unet_backbone_key = model_cfg.get("unet_backbone", "unet_res")
        if unet_backbone_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown UNet backbone '{unet_backbone_key}'")
        unet_cfg = MODEL_CONFIGS[unet_backbone_key].copy()
        unet = build_network(unet_cfg, in_channels, image_size, lr_channels, n_steps=None).to(device)
    
    # 2. 构建 I2SB（用于桥接）
    i2sb_backbone_key = model_cfg.get("i2sb_backbone", "unet_res")
    if i2sb_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown I2SB backbone '{i2sb_backbone_key}'")
    i2sb_cfg = MODEL_CONFIGS[i2sb_backbone_key].copy()
    n_steps = model_cfg["diffusion_steps"]
    
    # I2SB 输入是：粗糙HR(1) + TFM(3) = 4通道条件
    i2sb_lr_channels = in_channels + lr_channels if two_stage else lr_channels
    i2sb_net = build_network(i2sb_cfg,
                             in_channels,
                             image_size,
                             i2sb_lr_channels,
                             n_steps,
                             noise_levels=noise_levels).to(device)
    
    return unet, i2sb_net


def maybe_load_checkpoint(unet: Optional[nn.Module],
                          i2sb_net: nn.Module,
                          cfg: Dict[str, Any],
                          device: torch.device,
                          two_stage: bool = True) -> None:
    """加载预训练模型权重。"""
    model_cfg = cfg["model"]
    
    if two_stage:
        # 加载 UNet 权重（必须，用于生成粗糙 HR）
        unet_ckpt_path = model_cfg.get("unet_checkpoint")
        if not unet_ckpt_path:
            raise ValueError("unet_checkpoint is required for two-stage training")
        
        print(f"Loading UNet from {unet_ckpt_path}")
        unet_checkpoint = torch.load(unet_ckpt_path, map_location=device)
        unet_state_dict = select_state_dict(unet_checkpoint)
        if unet is None:
            raise ValueError("UNet model is not initialized.")
        unet.load_state_dict(unet_state_dict)
        unet.eval()  # UNet 冻结，不训练
        for param in unet.parameters():
            param.requires_grad = False
        print("UNet loaded and frozen")
    
    # 加载 I2SB 权重（可选，用于继续训练）
    i2sb_ckpt_path = model_cfg.get("i2sb_checkpoint")
    if i2sb_ckpt_path:
        print(f"Loading I2SB from {i2sb_ckpt_path}")
        i2sb_checkpoint = torch.load(i2sb_ckpt_path, map_location=device)
        i2sb_state_dict = select_state_dict(i2sb_checkpoint)
        i2sb_net.load_state_dict(i2sb_state_dict)
        print("I2SB checkpoint loaded")


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

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

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


def train(diffusion: Diffusion,
          unet: Optional[nn.Module],
          i2sb_net: nn.Module,
          cfg: Dict[str, Any],
          device: torch.device,
          ckpt_path: Path,
          log_dir: Path,
          n_steps: int,
          log: Logger,
          two_stage: bool = True) -> nn.Module:
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimization"]
    logging_cfg = cfg["logging"]
    stage_desc = "two-stage I2SB with frozen UNet backbone" if two_stage else "single-stage I2SB (no UNet)"
    log.info(f"Training {stage_desc}.")

    writer = SummaryWriter(log_dir=str(log_dir))
    
    train_loader, val_loader = create_train_val_loaders(cfg)
    total_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset) if val_loader is not None else 0
    
    # 获取实际图像尺寸
    sample_lr, sample_hr, _ = next(iter(train_loader))
    actual_image_size = sample_hr.shape[-1]
    
    log.info(f"Start training, batch size: {data_cfg['batch_size']}, epochs: {opt_cfg['epochs']}")
    if val_loader is None:
        log.info(f"Total samples: {total_samples}, steps per epoch: {len(train_loader)}, image size: {actual_image_size}x{actual_image_size}, HR channels: {sample_hr.shape[1]}, LR channels: {sample_lr.shape[1]}")
    else:
        log.info(f"Total samples: {total_samples}, val samples: {val_samples}, steps per epoch: {len(train_loader)}, image size: {actual_image_size}x{actual_image_size}, HR channels: {sample_hr.shape[1]}, LR channels: {sample_lr.shape[1]}")
    
    # UNet 已经冻结，只训练 I2SB
    if two_stage:
        if unet is None:
            raise ValueError("UNet model is required for two-stage training.")
        unet.eval()
    
    # 多GPU支持
    use_multi_gpu = cfg.get("multi_gpu", False)
    if use_multi_gpu and torch.cuda.device_count() > 1:
        log.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        i2sb_net = nn.DataParallel(i2sb_net)
        if two_stage and unet is not None:
            unet = nn.DataParallel(unet)
    
    i2sb_net = i2sb_net.to(device).train()
    ema_net = deepcopy(i2sb_net).eval().requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        i2sb_net.parameters(),  # 只优化 I2SB
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
    steps = space_indices(n_steps, n_steps)
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
        with log.progress_bar(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for tfm, hr_images, _ in pbar:
                tfm = tfm.to(device, non_blocking=True)  # TFM 3通道
                hr_images = hr_images.to(device, non_blocking=True)
                batch_size = hr_images.size(0)

                if two_stage:
                    # 第一级：UNet 生成粗糙 HR（冻结）
                    with torch.inference_mode():
                        coarse_hr = unet(tfm)  # [B, 1, H, W]
                    # 构建条件输入：粗糙HR(1) + TFM(3) = 4通道
                    condition = torch.cat([coarse_hr, tfm], dim=1)  # [B, 4, H, W]
                    bridge_start = coarse_hr
                else:
                    condition = tfm
                    bridge_start = tfm[:, :1] if tfm.shape[1] > 1 else tfm

                # 第二级：I2SB 在粗糙HR和真实HR之间构建桥
                t = torch.randint(0,
                                  n_steps, (batch_size,),
                                  device=device,
                                  dtype=torch.long)
                # 从起始到真实HR的桥
                x_t = diffusion.q_sample(t, hr_images, bridge_start)

                with torch.amp.autocast(device_type=device.type,
                                        dtype=amp_dtype,
                                        enabled=use_amp
                                        and device.type == "cuda"):
                    pred = i2sb_net(x_t, t, condition)
                    std_fwd = diffusion.get_std_fwd(t, xdim=hr_images.shape[1:])
                    label = (x_t - hr_images) / std_fwd
                    loss = loss_fn(pred, label)

                optimizer.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(i2sb_net.parameters(),
                                                       clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(i2sb_net.parameters(),
                                                       clip_grad_norm)
                    optimizer.step()

                total_loss += loss.item() * batch_size
                ema_update(ema_net, i2sb_net, decay=opt_cfg["ema_decay"])
                pbar.update_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = None
        if val_loader is not None:
            net_was_training = i2sb_net.training
            i2sb_net.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for val_tfm, val_hr_images, _ in val_loader:
                    val_tfm = val_tfm.to(device, non_blocking=True)
                    val_hr_images = val_hr_images.to(device, non_blocking=True)
                    batch_size = val_hr_images.size(0)

                    if two_stage:
                        coarse_hr = unet(val_tfm)
                        condition = torch.cat([coarse_hr, val_tfm], dim=1)
                        bridge_start = coarse_hr
                    else:
                        condition = val_tfm
                        bridge_start = val_tfm[:, :1] if val_tfm.shape[1] > 1 else val_tfm

                    t = torch.randint(0,
                                      n_steps, (batch_size,),
                                      device=device,
                                      dtype=torch.long)
                    x_t = diffusion.q_sample(t, val_hr_images, bridge_start)

                    with torch.amp.autocast(device_type=device.type,
                                            dtype=amp_dtype,
                                            enabled=use_amp
                                            and device.type == "cuda"):
                        pred = i2sb_net(x_t, t, condition)
                        std_fwd = diffusion.get_std_fwd(t, xdim=val_hr_images.shape[1:])
                        label = (x_t - val_hr_images) / std_fwd
                        loss = loss_fn(pred, label)

                    val_loss += loss.item() * batch_size
            avg_val_loss = val_loss / len(val_loader.dataset)
            writer.add_scalar('val/loss', avg_val_loss, epoch + 1)
            if net_was_training:
                i2sb_net.train()

        if epoch % preview_interval == 0:
            i2sb_net_was_training = i2sb_net.training
            i2sb_net.eval()
            ema_net.eval()
            with torch.inference_mode():
                preview_batch = min(preview_count, tfm.size(0))
                tfm_subset = tfm[:preview_batch]
                if two_stage:
                    coarse_subset = unet(tfm_subset)  # 粗糙HR
                    condition_subset = torch.cat([coarse_subset, tfm_subset], dim=1)
                else:
                    coarse_subset = None
                    condition_subset = tfm_subset
                img_net = diffusion.ddpm_sampling(steps, i2sb_net, condition_subset).cpu()
                img_ema = diffusion.ddpm_sampling(steps, ema_net, condition_subset).cpu()
            if i2sb_net_was_training:
                i2sb_net.train()

            hr_subset = hr_images[:preview_batch].cpu()
            hr01 = ((hr_subset.clamp(-1, 1) + 1) / 2)
            net01 = ((img_net.detach().cpu().clamp(-1, 1) + 1) / 2)
            ema01 = ((img_ema.detach().cpu().clamp(-1, 1) + 1) / 2)
            channels = hr01.shape[1]
            # writer.add_image(f'sample/epoch_{epoch + 1}_tfm',
            #                  make_preview_grid(tfm01, 1, preview_nrow),
            #                  epoch + 1)
            if two_stage and coarse_subset is not None:
                coarse_subset_vis = coarse_subset.cpu()
                coarse01 = ((coarse_subset_vis.clamp(-1, 1) + 1) / 2)
                writer.add_image(f'sample/epoch_{epoch + 1}_coarse_hr',
                                 make_preview_grid(coarse01, channels, preview_nrow),
                                 epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_hr',
                             make_preview_grid(hr01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_i2sb_net',
                             make_preview_grid(net01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_i2sb_ema',
                             make_preview_grid(ema01, channels, preview_nrow),
                             epoch + 1)

        avg_loss = total_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/loss', avg_loss, epoch + 1)
        writer.add_scalar('train/learning_rate', current_lr, epoch + 1)

        toc = time.time()
        if avg_val_loss is None:
            log.info(
                f"Epoch {epoch + 1}/{epochs} finished. "
                f"Average loss: {avg_loss:.6f}. "
                f"LR: {current_lr:.2e}")
        else:
            log.info(
                f"Epoch {epoch + 1}/{epochs} finished. "
                f"Train loss: {avg_loss:.6f}. "
                f"Val loss: {avg_val_loss:.6f}. "
                f"LR: {current_lr:.2e}")

        metric_loss = avg_val_loss if avg_val_loss is not None else avg_loss
        if metric_loss < best_loss:
            best_loss = metric_loss
            best_loss_epoch = epoch + 1
            # 保存时去除DataParallel的module.前缀
            model_state = i2sb_net.module.state_dict() if isinstance(i2sb_net, nn.DataParallel) else i2sb_net.state_dict()
            ema_state = ema_net.module.state_dict() if isinstance(ema_net, nn.DataParallel) else ema_net.state_dict()
            model_best_state_dict = deepcopy(model_state)
            ema_model_best_state_dict = deepcopy(ema_state)

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

        # 保存时去除DataParallel的module.前缀
        model_state = i2sb_net.module.state_dict() if isinstance(i2sb_net, nn.DataParallel) else i2sb_net.state_dict()
        ema_state = ema_net.module.state_dict() if isinstance(ema_net, nn.DataParallel) else ema_net.state_dict()
        
        ckpt = {
            'epoch': epoch + 1,
            'ema_decay': opt_cfg["ema_decay"],
            'model_state_dict': model_state,
            'ema_model_state_dict': ema_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_loss_epoch': best_loss_epoch,
            'model_best_state_dict': model_best_state_dict,
            'ema_model_best_state_dict': ema_model_best_state_dict,
        }
        tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(ckpt, tmp_ckpt)
        safe_replace(tmp_ckpt, ckpt_path)

    writer.close()
    log.info("Training completed!")
    return ema_net


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def main() -> None:
    # 可选择输入一段记录日志
    user_log = input("请输入一段日志记录（或直接回车跳过）: ")
    
    # 初始化 logger
    log = Logger(rank=0, log_dir="runs/logs")
    
    single_stage = env_flag("SR_SINGLE_STAGE")
    title = "Single-Stage I2SB (no UNet)" if single_stage else "Two-Stage I2SB: UNet -> Schrodinger Bridge Refinement"
    log.info("=======================================================")
    log.info(f"   {title}")
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

    n_steps = cfg["model"]["diffusion_steps"]
    if cfg["data"].get("augment", False):
        log.info("数据增强已启用")   
    if cfg["data"].get("use_tfm_channels", False):
        log.info("使用 TFM 3通道作为输入")
    else:
        log.info("使用单通道作为输入")
    log.info(f"Diffusion steps: {n_steps}")

    # 对称 beta 调度（I2SB 桥接）
    betas = make_beta_schedule(n_timestep=n_steps, linear_end=3e-4)
    half = n_steps // 2
    if n_steps % 2 == 1:
        # 奇数步数：中间点重复一次
        betas = np.concatenate([betas[:half], [betas[half]], np.flip(betas[:half])])
    else:
        # 偶数步数：直接镜像
        betas = np.concatenate([betas[:half], np.flip(betas[:half])])
    diffusion = Diffusion(betas, device)

    # 计算noise_levels（I2SB的时间编码）
    t0 = float(cfg["model"].get("t0", 1e-4))
    T = float(cfg["model"].get("T", 1.0))
    noise_levels = torch.linspace(t0, T, n_steps, device=device, dtype=torch.float32) * n_steps
    log.info(f"Noise levels range: [{noise_levels.min().item():.4g}, {noise_levels.max().item():.4g}]")
    
    two_stage = not single_stage
    # 构建两个模型：UNet 和 I2SB
    unet, i2sb_net = build_models(cfg, device, noise_levels=noise_levels, two_stage=two_stage)
    maybe_load_checkpoint(unet, i2sb_net, cfg, device, two_stage=two_stage)
    if two_stage:
        log.info("Initialized two-stage model: frozen UNet + I2SB.")
    else:
        log.info("Initialized single-stage model: I2SB conditioned on LR.")

    ckpt_dir = Path(cfg["model"]["checkpoint_dir"])
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / cfg["model"]["checkpoint_name"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_cfg = cfg["logging"]
    log_root = Path(log_cfg["tensorboard_root"])
    ensure_dir(log_root)
    log_dir = log_root / f"{timestamp}-{log_cfg.get('experiment_name', 'i2sb-train')}"
    ensure_dir(log_dir)
    
    train(diffusion, unet, i2sb_net, cfg, device, ckpt_path, log_dir, n_steps, log, two_stage=two_stage)
    
    # 训练结束后保存配置文件（记录实际完成的训练）
    cfg_copy_path = log_dir / "train_config.json"
    with cfg_copy_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)
    log.info(f"Configuration saved to {cfg_copy_path}")


if __name__ == "__main__":
    main()
