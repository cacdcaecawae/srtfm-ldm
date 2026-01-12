import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.functional import (peak_signal_noise_ratio,
                                     structural_similarity_index_measure)
from matplotlib import cm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import get_h5_dataloader
from I2sb.diffusion import Diffusion
from network.networksb import (build_network, convnet_big_cfg, convnet_medium_cfg,
                     convnet_small_cfg, unet_res_cfg,
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
    "unet_res": unet_res_cfg,
    "unet_res_diffusion": unet_res_diffusion_cfg,
}

DEFAULT_CONFIG_PATH = Path(__file__).with_name("eval.json")


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


def env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader:
    """创建数据加载器"""
    coord_range = None
    if cfg.get("use_tfm_channels", False):
        coord_range_x = tuple(cfg["coord_range_x"]) if "coord_range_x" in cfg else (-1.0, 1.0)
        coord_range_y = tuple(cfg["coord_range_y"]) if "coord_range_y" in cfg else (-1.0, 1.0)
        coord_range = (coord_range_x, coord_range_y)
    
    return get_h5_dataloader(
        h5_path=cfg["h5_path"],
        batch_size=cfg["batch_size"],
        lr_key=cfg.get("h5_lr_key", "TFM"),
        hr_key=cfg.get("h5_hr_key", "hr"),
        lr_dataset_name=cfg.get("h5_lr_dataset"),
        hr_dataset_name=cfg.get("h5_hr_dataset"),
        transpose_lr=cfg.get("transpose_lr", False),
        transpose_hr=cfg.get("transpose_hr", False),
        use_tfm_channels=cfg.get("use_tfm_channels", False),
        coord_range=coord_range,
        augment=cfg.get("augment", False),
        h_flip_prob=cfg.get("h_flip_prob", 0.0),
        translate_prob=cfg.get("translate_prob", 0.0),
        max_translate_ratio=cfg.get("max_translate_ratio", 0.0),
        num_workers=cfg.get("num_workers", 4),
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


def build_models(cfg: Dict[str, Any],
                 device: torch.device,
                 two_stage: bool = True) -> Tuple[Optional[torch.nn.Module], torch.nn.Module]:
    """Build the frozen UNet and the I2SB diffusion model."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    i2sb_backbone_key = model_cfg.get("i2sb_backbone",
                                      model_cfg.get("backbone", "unet_res_diffusion"))
    if i2sb_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown I2SB backbone '{i2sb_backbone_key}'. "
                       f"Available: {', '.join(MODEL_CONFIGS)}")

    n_steps = model_cfg["diffusion_steps"]
    t0 = float(model_cfg.get("t0", 1e-4))
    T = float(model_cfg.get("T", 1.0))
    noise_levels = torch.linspace(t0, T, n_steps, device=device, dtype=torch.float32) * n_steps

    in_channels = data_cfg["channels"]
    image_size = data_cfg["image_size"]

    lr_channels = 3 if data_cfg.get("use_tfm_channels", False) else in_channels
    i2sb_lr_channels = in_channels + lr_channels if two_stage else lr_channels
    unet = None

    i2sb_cfg = MODEL_CONFIGS[i2sb_backbone_key].copy()
    i2sb_net = build_network(i2sb_cfg,
                             in_channels=in_channels,
                             image_size=image_size,
                             lr_channels=i2sb_lr_channels,
                             n_steps=n_steps,
                             noise_levels=noise_levels).to(device)

    i2sb_ckpt_path = model_cfg.get("checkpoint_path")
    if not i2sb_ckpt_path:
        raise ValueError("checkpoint_path is required for I2SB evaluation.")
    i2sb_checkpoint = torch.load(i2sb_ckpt_path, map_location=device)
    i2sb_state_dict = select_state_dict(i2sb_checkpoint)
    i2sb_net.load_state_dict(i2sb_state_dict)
    i2sb_net.eval()
    print(f"Loaded I2SB model from {i2sb_ckpt_path}")

    if two_stage:
        unet_backbone_key = model_cfg.get("unet_backbone", "unet_res")
        if unet_backbone_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown UNet backbone '{unet_backbone_key}'. "
                           f"Available: {', '.join(MODEL_CONFIGS)}")
        unet_ckpt_path = model_cfg.get("unet_checkpoint")
        if not unet_ckpt_path:
            raise ValueError("unet_checkpoint is required for two-stage I2SB evaluation.")
        unet_cfg = MODEL_CONFIGS[unet_backbone_key].copy()
        unet = build_network(unet_cfg,
                             in_channels=in_channels,
                             image_size=image_size,
                             lr_channels=lr_channels,
                             n_steps=None).to(device)
        unet_checkpoint = torch.load(unet_ckpt_path, map_location=device)
        unet_state_dict = select_state_dict(unet_checkpoint)
        unet.load_state_dict(unet_state_dict)
        unet.eval()
        print(f"Loaded UNet model from {unet_ckpt_path}")

    return unet, i2sb_net


def make_beta_schedule(n_timestep: int = 1000,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2) -> np.ndarray:
    """创建 beta schedule（与训练代码一致）"""
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
                      n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def create_diffusion(cfg: Dict[str, Any], device: torch.device) -> Diffusion:
    """创建 I2SB 扩散对象（与训练代码保持一致）"""
    n_steps = cfg["model"]["diffusion_steps"]
    
    # 使用与训练相同的 beta schedule
    betas = make_beta_schedule(n_timestep=n_steps, linear_end=3e-4)
    
    # 对称镜像处理（I2SB 桥接）
    half = n_steps // 2
    if n_steps % 2 == 1:
        # 奇数步数：中间点重复一次
        betas = np.concatenate([betas[:half], [betas[half]], np.flip(betas[:half])])
    else:
        # 偶数步数：直接镜像
        betas = np.concatenate([betas[:half], np.flip(betas[:half])])
    
    diffusion = Diffusion(betas, device)
    return diffusion


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """反归一化到 [0, 1]"""
    return tensor.clamp(-1, 1).add(1).div(2)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测张量 [C, H, W] 或 [B, C, H, W]，值域 [0, 1]
        target: 目标张量，形状与pred相同
        threshold: 二值化阈值
    
    Returns:
        IoU值
    """
    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算交集和并集
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # 避免除零
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def tensor_to_image(tensor: torch.Tensor,
                    apply_jet: bool = False) -> Image.Image:
    """将张量转换为 PIL Image"""
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
        return Image.fromarray(array[:, :, 0], mode='L')
    return Image.fromarray(array)


def evaluate(cfg: Dict[str, Any], device: torch.device, two_stage: bool = True) -> None:
    """执行 I2SB 评估"""
    data_cfg = cfg["data"]
    sampler_cfg = cfg.get("sampler", {})
    
    dataloader = create_dataloader(data_cfg)
    unet, net = build_models(cfg, device, two_stage=two_stage)
    diffusion = create_diffusion(cfg, device)
    
    use_jet = data_cfg.get("channels", 1) == 1
    
    output_root = Path(cfg["output"]["root"])
    images_dir = output_root / "images"
    ensure_dir(images_dir)
    ensure_dir(output_root)
    
    # I2SB 采样参数
    num_steps = sampler_cfg.get("num_steps", 100)
    ot_ode = sampler_cfg.get("ot_ode", False)
    n_diffusion_steps = cfg["model"]["diffusion_steps"]
    
    # 生成时间步序列 [0, step_size, ..., n_diffusion_steps-1]
    # 注意：步数范围是 0 到 n_diffusion_steps-1，因为嵌入层索引从 0 开始
    step_size = n_diffusion_steps // num_steps
    steps = np.arange(0, n_diffusion_steps, step_size)
    if steps[-1] != n_diffusion_steps - 1:
        steps = np.append(steps, n_diffusion_steps - 1)
    
    print(f"I2SB Sampling: {num_steps} steps, OT-ODE={ot_ode}")
    
    psnr_scores = []
    ssim_scores = []
    iou_scores = []
    per_image_results = []
    
    with torch.inference_mode():
        for lr_images, hr_images, names in tqdm(dataloader, desc="Evaluating"):
            # 固定随机种子以保证可重复性
            seed = sampler_cfg.get("seed", 1234)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            if two_stage:
                coarse_hr = unet(lr_images)
                condition = torch.cat([coarse_hr, lr_images], dim=1)
            else:
                condition = lr_images

            # I2SB 采样：使用 ddpm_sampling 方法
            sr_images = diffusion.ddpm_sampling(
                steps=steps,
                net=net,
                x1=condition,
                ot_ode=ot_ode,
                verbose=False
            )
            
            # 反归一化
            sr_for_metric = denormalize(sr_images)
            
            # 阈值处理
            threshold = sampler_cfg.get("threshold", 0.05)
            sr_for_metric = torch.where(
                sr_for_metric < threshold,
                torch.zeros_like(sr_for_metric),
                sr_for_metric
            )
            
            # 压缩到 [0, 227/253]
            sr_for_image = sr_for_metric #* (227.0 / 253.0)
            
            hr_for_metric = denormalize(hr_images)
            
            # 计算指标
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
                
                # 保存图像
                if cfg["output"].get("save_images", True):
                    if not any(name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                        image_filename = f"{name}.png"
                    else:
                        image_filename = name
                    tensor_to_image(sr_for_image[idx],
                                    apply_jet=use_jet).save(images_dir / image_filename)
    
    # 计算统计指标（平均、最大、最小）
    avg_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    max_psnr = float(np.max(psnr_scores)) if psnr_scores else 0.0
    min_psnr = float(np.min(psnr_scores)) if psnr_scores else 0.0
    
    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    max_ssim = float(np.max(ssim_scores)) if ssim_scores else 0.0
    min_ssim = float(np.min(ssim_scores)) if ssim_scores else 0.0
    
    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    max_iou = float(np.max(iou_scores)) if iou_scores else 0.0
    min_iou = float(np.min(iou_scores)) if iou_scores else 0.0
    
    print(f"PSNR - 平均: {avg_psnr:.4f}, 最大: {max_psnr:.4f}, 最小: {min_psnr:.4f}")
    print(f"SSIM - 平均: {avg_ssim:.4f}, 最大: {max_ssim:.4f}, 最小: {min_ssim:.4f}")
    print(f"IoU  - 平均: {avg_iou:.4f}, 最大: {max_iou:.4f}, 最小: {min_iou:.4f}")
    
    # 保存结果
    results_txt = output_root / "results.txt"
    data_source = cfg["data"].get("h5_path", "N/A")
    with results_txt.open("w", encoding="utf-8") as handle:
        handle.write("I2SB Evaluation Summary\n")
        handle.write("======================\n")
        handle.write(f"Model: {cfg['model']['checkpoint_path']}\n")
        handle.write(f"Dataset: {data_source}\n")
        handle.write(f"Sampling Steps: {num_steps}\n")
        handle.write(f"OT-ODE: {ot_ode}\n")
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
    parser = argparse.ArgumentParser(description="Evaluate I2SB diffusion model.")
    parser.add_argument("--config",
                        type=Path,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to I2SB evaluation configuration JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")
    single_stage = env_flag("SR_SINGLE_STAGE")
    if single_stage:
        print("Single-stage mode: I2SB conditioned on LR only.")
    evaluate(cfg, device, two_stage=not single_stage)


if __name__ == "__main__":
    main()
   # python I2sb/eval.py --config I2sb/eval.json
