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
from network.network import build_network, unet_res_cfg


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "unet_res": unet_res_cfg,
}

DEFAULT_CONFIG_PATH = Path("train.json")


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


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    backbone_key = model_cfg["backbone"]
    if backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown backbone '{backbone_key}'. "
                       f"Available: {', '.join(MODEL_CONFIGS)}")
    net_cfg = MODEL_CONFIGS[backbone_key].copy()
    
    # 获取输入通道数和图像尺寸
    in_channels = data_cfg["channels"]  # HR 的通道数
    image_size = data_cfg["image_size"]
    
    # LR 的通道数
    lr_channels = in_channels
    if data_cfg.get("use_tfm_channels", False):
        lr_channels = 3  # TFM 模式: I, X, Y 三通道

    net = build_network(net_cfg,
                        in_channels=in_channels,
                        image_size=image_size,
                        lr_channels=lr_channels).to(device)
    return net


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


def maybe_load_checkpoint(net: nn.Module, cfg: Dict[str, Any],
                          device: torch.device) -> None:
    resume_path = cfg["model"].get("resume_checkpoint")
    if not resume_path:
        return
    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = select_state_dict(checkpoint)
    net.load_state_dict(state_dict)
    # 注意：这里不使用 log，因为这个函数可能在 log 初始化前调用
    print(f"Loaded weights from {resume_path}")


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

def train(net: nn.Module,
          cfg: Dict[str, Any],
          device: torch.device,
          ckpt_path: Path,
          log_dir: Path,
          log: Logger) -> nn.Module:
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimization"]
    logging_cfg = cfg["logging"]
    log.info("Training direct UNet with pixel-wise MSE loss.")

    writer = SummaryWriter(log_dir=str(log_dir))
    
    dataloader = create_dataloader(cfg)
    total_samples = len(dataloader.dataset)
    
    # 获取实际图像尺寸
    sample_lr, sample_hr, _ = next(iter(dataloader))
    actual_image_size = sample_hr.shape[-1]
    
    log.info(f"Start training, batch size: {data_cfg['batch_size']}, epochs: {opt_cfg['epochs']}")
    log.info(f"Total samples: {total_samples}, steps per epoch: {len(dataloader)}, image size: {actual_image_size}x{actual_image_size}, HR channels: {sample_hr.shape[1]}, LR channels: {sample_lr.shape[1]}")
    net = net.to(device).train()
    ema_net = deepcopy(net).eval().requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        net.parameters(),
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

                with torch.amp.autocast(device_type=device.type,
                                        dtype=amp_dtype,
                                        enabled=use_amp
                                        and device.type == "cuda"):
                    pred = net(lr_images)
                    loss = loss_fn(pred, hr_images)

                optimizer.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                       clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                       clip_grad_norm)
                    optimizer.step()

                total_loss += loss.item() * batch_size
                ema_update(ema_net, net, decay=opt_cfg["ema_decay"])
                pbar.update_postfix(loss=f"{loss.item():.4f}")

        if epoch % preview_interval == 0:
            net_was_training = net.training
            net.eval()
            ema_net.eval()
            with torch.inference_mode():
                preview_batch = min(preview_count, lr_images.size(0))
                lr_subset = lr_images[:preview_batch]
                img_net = net(lr_subset).detach().cpu()
                img_ema = ema_net(lr_subset).detach().cpu()
            if net_was_training:
                net.train()

            hr_subset = hr_images[:preview_batch].cpu()
            lr01 = ((lr_subset.detach().cpu().clamp(-1, 1) + 1) / 2)
            hr01 = ((hr_subset.clamp(-1, 1) + 1) / 2)
            net01 = ((img_net.detach().cpu().clamp(-1, 1) + 1) / 2)
            ema01 = ((img_ema.detach().cpu().clamp(-1, 1) + 1) / 2)
            channels = lr01.shape[1]
            writer.add_image(f'sample/epoch_{epoch + 1}_lr',
                             make_preview_grid(lr01, channels, preview_nrow),
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
            model_best_state_dict = deepcopy(net.state_dict())
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
            'model_state_dict': net.state_dict(),
            'ema_model_state_dict': ema_net.state_dict(),
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


def collect_preview_batch(cfg: Dict[str, Any],
                          device: torch.device,
                          count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    dataloader = create_dataloader(cfg)
    lr_images, hr_images, _ = next(iter(dataloader))
    lr_images = lr_images[:count].to(device)
    hr_images = hr_images[:count].to(device)
    return lr_images, hr_images


def main() -> None:
    # 可选择输入一段记录日志
    user_log = input("请输入一段日志记录（或直接回车跳过）: ")
    
    # 初始化 logger
    log = Logger(rank=0, log_dir="runs/logs")
    
    log.info("=======================================================")
    log.info("           UNetonly Super-Resolution Trainer")
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

    net = build_model(cfg, device)
    maybe_load_checkpoint(net, cfg, device)
    log.info("Initialized UNet for direct SR training.")

    ckpt_dir = Path(cfg["model"]["checkpoint_dir"])
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / cfg["model"]["checkpoint_name"]
    timestamp = time.strftime("%Y%m%d-%H%M%S") #时间戳获取
    log_cfg = cfg["logging"]
    log_root = Path(log_cfg["tensorboard_root"])
    ensure_dir(log_root)
    log_dir = log_root / f"{timestamp}-{log_cfg.get('experiment_name', 'sr-train')}"
    ensure_dir(log_dir)
    
    train(net, cfg, device, ckpt_path, log_dir, log)
    
    # 训练结束后保存配置文件（记录实际完成的训练）
    cfg_copy_path = log_dir / "train_config.json"
    with cfg_copy_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)
    log.info(f"Configuration saved to {cfg_copy_path}")


if __name__ == "__main__":
    main()
