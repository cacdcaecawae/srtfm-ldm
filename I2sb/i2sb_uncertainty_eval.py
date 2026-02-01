"""
I2SB 不确定性量化评估脚本

对每个测试样本进行多次独立采样（默认5次），计算像素级不确定性 U_pixel，
并根据不确定性水平分组统计性能指标。

像素级不确定性定义：
    U_pixel = (1 / HW) * Σ_x Σ_y Var({M^(k)_{y,x}}_{k=1}^K)

分组标准：
    - 高置信度组：U_pixel < 0.05
    - 低置信度组：U_pixel > 0.15
"""

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
from scipy.ndimage import distance_transform_edt
import cv2

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

DEFAULT_CONFIG_PATH = Path(__file__).with_name("uncertainty_eval.json")


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
                             lr_channels=1,
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


def compute_pixel_uncertainty(samples: torch.Tensor) -> float:
    """
    计算像素级不确定性 U_pixel
    
    Args:
        samples: [K, C, H, W] 张量，K次采样结果
    
    Returns:
        U_pixel: 像素级不确定性（方差的全图平均）
    """
    # 计算每个像素在K次采样中的方差
    pixel_variance = torch.var(samples, dim=0, unbiased=False)  # [C, H, W]
    # 对所有像素取平均
    u_pixel = pixel_variance.mean().item()
    return u_pixel


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算IoU (Intersection over Union)"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算Dice系数"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    if pred_sum + target_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + target_sum)
    return dice.item()


def compute_hd95(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, percentile: float = 95.0) -> float:
    """计算95th percentile Hausdorff Distance"""
    from scipy.ndimage import binary_erosion
    
    pred_np = (pred > threshold).cpu().numpy().astype(bool)
    target_np = (target > threshold).cpu().numpy().astype(bool)

    while pred_np.ndim > 2:
        pred_np = pred_np.any(axis=0)
    while target_np.ndim > 2:
        target_np = target_np.any(axis=0)
    
    if not pred_np.any() and not target_np.any():
        return 0.0
    
    if not pred_np.any() or not target_np.any():
        h, w = pred_np.shape
        return float(np.sqrt(h**2 + w**2))

    pred_dt = distance_transform_edt(~pred_np) 
    target_dt = distance_transform_edt(~target_np)

    pred_border = pred_np ^ binary_erosion(pred_np, border_value=0)
    target_border = target_np ^ binary_erosion(target_np, border_value=0)
    
    if not pred_border.any(): 
        pred_border = pred_np
    if not target_border.any(): 
        target_border = target_np

    d_pred_to_target = target_dt[pred_border]
    d_target_to_pred = pred_dt[target_border]
    
    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    
    if len(all_distances) == 0:
        return 0.0
        
    return float(np.percentile(all_distances, percentile))


def tensor_to_image(tensor: torch.Tensor, apply_jet: bool = False) -> Image.Image:
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


def overlay_contour(pred_tensor: torch.Tensor, 
                    gt_tensor: torch.Tensor,
                    apply_jet: bool = False,
                    threshold: float = 0.5,
                    contour_color: tuple = (0, 255, 0)) -> Image.Image:
    """在预测图像上叠加GT轮廓"""
    pred_array = pred_tensor.permute(1, 2, 0).cpu().numpy()
    pred_array = np.clip(pred_array, 0.0, 1.0)
    
    gt_array = gt_tensor.cpu().numpy()
    if gt_array.ndim == 3:
        gt_array = gt_array.max(axis=0)
    gt_binary = (gt_array > threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if apply_jet:
        if pred_array.shape[2] == 1:
            gray_np = pred_array[:, :, 0]
        else:
            gray_np = pred_array.mean(axis=2)
        gray_uint8 = (gray_np * 255.0).astype(np.uint8)
        img_rgb = cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    else:
        pred_uint8 = (pred_array * 255.0).astype(np.uint8)
        if pred_uint8.shape[2] == 1:
            img_rgb = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = pred_uint8
    
    cv2.drawContours(img_rgb, contours, -1, contour_color, thickness=1)
    
    return Image.fromarray(img_rgb)


def evaluate_with_uncertainty(cfg: Dict[str, Any], device: torch.device, two_stage: bool = True) -> None:
    """执行不确定性量化评估"""
    data_cfg = cfg["data"]
    sampler_cfg = cfg.get("sampler", {})
    uncertainty_cfg = cfg.get("uncertainty", {})
    
    dataloader = create_dataloader(data_cfg)
    unet, net = build_models(cfg, device, two_stage=two_stage)
    diffusion = create_diffusion(cfg, device)
    
    use_jet = data_cfg.get("channels", 1) == 1
    
    output_root = Path(cfg["output"]["root"])
    images_dir = output_root / "images"
    variance_dir = output_root / "variance"
    uncertainty_dir = output_root / "uncertainty_maps"
    ensure_dir(images_dir)
    ensure_dir(variance_dir)
    ensure_dir(uncertainty_dir)
    ensure_dir(output_root)
    
    # 不确定性量化参数
    n_samples = uncertainty_cfg.get("n_samples", 5)  # 每个样本采样5次
    low_threshold = uncertainty_cfg.get("low_threshold", 0.05)  # 高置信度阈值
    high_threshold = uncertainty_cfg.get("high_threshold", 0.15)  # 低置信度阈值
    
    # I2SB 采样参数
    num_steps = sampler_cfg.get("num_steps", 100)
    ot_ode = sampler_cfg.get("ot_ode", False)
    n_diffusion_steps = cfg["model"]["diffusion_steps"]
    threshold = sampler_cfg.get("threshold", 0.05)
    
    # 生成时间步序列
    step_size = n_diffusion_steps // num_steps
    steps = np.arange(0, n_diffusion_steps, step_size)
    if steps[-1] != n_diffusion_steps - 1:
        steps = np.append(steps, n_diffusion_steps - 1)
    
    print(f"不确定性量化评估配置：")
    print(f"  - 每样本采样次数：{n_samples}")
    print(f"  - 高置信度阈值：U_pixel < {low_threshold}")
    print(f"  - 低置信度阈值：U_pixel > {high_threshold}")
    print(f"  - I2SB采样步数：{num_steps}, OT-ODE={ot_ode}")
    print(f"  - 批次大小：{data_cfg['batch_size']}, 最多测试：{120}个样本")
    print(f"  - 批量并行：每批处理 {data_cfg['batch_size']} × {n_samples} = {data_cfg['batch_size'] * n_samples} 个推理")
    
    # 存储所有样本的结果
    all_results = []
    max_samples = 180  # 只测试180个样本
    sample_count = 0
    
    # 计算预期的batch数量
    batch_size = data_cfg['batch_size']
    expected_batches = (max_samples + batch_size - 1) // batch_size
    
    with torch.inference_mode():
        pbar = tqdm(total=max_samples, desc="不确定性评估", unit="样本")
        for lr_images, hr_images, names in dataloader:
            # 设置随机种子（用于可重复性）
            seed = sampler_cfg.get("seed", 1234)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            batch_size = lr_images.shape[0]
            
            # 准备condition（two-stage或single-stage）
            if two_stage:
                coarse_hr = unet(lr_images[:, :1])
                condition = torch.cat([coarse_hr, lr_images], dim=1)
            else:
                condition = lr_images
            
            # 🚀 批量并行处理：[B, C, H, W] -> [B*K, C, H, W]
            # 将batch中的每个样本复制K次，一次性处理所有
            condition_expanded = condition.repeat(n_samples, 1, 1, 1)  # [B*K, C, H, W]
            
            # I2SB 批量采样：一次性处理所有样本的所有采样
            sr_samples_all = diffusion.ddpm_sampling(
                steps=steps,
                net=net,
                x1=condition_expanded,
                ot_ode=ot_ode,
                verbose=False
            )  # [B*K, C, H, W]
            
            # 反归一化和阈值处理
            sr_samples_all = denormalize(sr_samples_all)
            sr_samples_all = torch.where(
                sr_samples_all < threshold,
                torch.zeros_like(sr_samples_all),
                sr_samples_all
            )
            
            # 重塑为 [K, B, C, H, W] 以便按样本分组
            sr_samples_reshaped = sr_samples_all.view(n_samples, batch_size, *sr_samples_all.shape[1:])
            
            # 对batch中的每个样本单独处理
            for idx_in_batch in range(batch_size):
                name = names[idx_in_batch]
                hr_single = hr_images[idx_in_batch:idx_in_batch+1]
                
                # 提取该样本的所有采样结果：[K, C, H, W]
                samples = sr_samples_reshaped[:, idx_in_batch, :, :, :]
                
                # 计算像素级不确定性
                u_pixel = compute_pixel_uncertainty(samples)
                
                # 计算平均预测
                sr_mean = samples.mean(dim=0, keepdim=True)  # [1, C, H, W]
                sr_variance = samples.var(dim=0, keepdim=True)  # [1, C, H, W]
                
                # 计算性能指标
                hr_denorm = denormalize(hr_single)
                
                psnr = peak_signal_noise_ratio(sr_mean, hr_denorm, data_range=1.0)
                ssim = structural_similarity_index_measure(sr_mean, hr_denorm, data_range=1.0)
                iou = compute_iou(sr_mean, hr_denorm, threshold=threshold)
                dice = compute_dice(sr_mean, hr_denorm, threshold=threshold)
                hd95 = compute_hd95(sr_mean, hr_denorm, threshold=threshold)
                
                # 保存结果
                result = {
                    "filename": name,
                    "u_pixel": u_pixel,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "iou": iou,
                    "dice": dice,
                    "hd95": hd95,
                }
                all_results.append(result)
                
                # 保存图像
                if cfg["output"].get("save_images", True):
                    if not any(name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                        image_filename = f"{name}.png"
                    else:
                        image_filename = name
                    
                    # 保存平均预测（带GT轮廓）
                    overlay_contour(sr_mean[0],
                                    hr_denorm[0],
                                    apply_jet=use_jet,
                                    threshold=threshold).save(images_dir / image_filename)
                    
                    # 保存方差图
                    var_normalized = sr_variance[0]
                    var_min = var_normalized.min()
                    var_max = var_normalized.max()
                    if var_max > var_min:
                        var_normalized = (var_normalized - var_min) / (var_max - var_min)
                    else:
                        var_normalized = torch.zeros_like(var_normalized)
                    tensor_to_image(var_normalized, apply_jet=True).save(variance_dir / image_filename)
                    
                    # 保存不确定性图（与方差图相同，但标注U_pixel值）
                    uncertainty_img = tensor_to_image(var_normalized, apply_jet=True)
                    uncertainty_img.save(uncertainty_dir / image_filename)
                
                # 更新样本计数和进度条
                sample_count += 1
                pbar.update(1)
            
            # 达到120个样本后停止
            if sample_count >= max_samples:
                break
        
        pbar.close()
    
    # ========== 分组统计 ==========
    high_confidence = [r for r in all_results if r["u_pixel"] < low_threshold]
    low_confidence = [r for r in all_results if r["u_pixel"] > high_threshold]
    middle_confidence = [r for r in all_results if low_threshold <= r["u_pixel"] <= high_threshold]
    
    print(f"\n样本分组统计：")
    print(f"  - 高置信度组（U_pixel < {low_threshold}）：{len(high_confidence)} 个样本")
    print(f"  - 低置信度组（U_pixel > {high_threshold}）：{len(low_confidence)} 个样本")
    print(f"  - 中等置信度组：{len(middle_confidence)} 个样本")
    print(f"  - 总样本数：{len(all_results)}")
    
    def compute_group_stats(group: List[Dict[str, Any]], group_name: str) -> Dict[str, float]:
        """计算分组统计"""
        if not group:
            return {
                "count": 0,
                "avg_u_pixel": 0.0,
                "avg_psnr": 0.0,
                "avg_ssim": 0.0,
                "avg_iou": 0.0,
                "avg_dice": 0.0,
                "avg_hd95": 0.0,
            }
        
        stats = {
            "count": len(group),
            "avg_u_pixel": np.mean([r["u_pixel"] for r in group]),
            "avg_psnr": np.mean([r["psnr"] for r in group]),
            "avg_ssim": np.mean([r["ssim"] for r in group]),
            "avg_iou": np.mean([r["iou"] for r in group]),
            "avg_dice": np.mean([r["dice"] for r in group]),
            "avg_hd95": np.mean([r["hd95"] for r in group]),
        }
        
        print(f"\n{group_name} (n={stats['count']}):")
        print(f"  平均 U_pixel: {stats['avg_u_pixel']:.6f}")
        print(f"  平均 PSNR: {stats['avg_psnr']:.4f}")
        print(f"  平均 SSIM: {stats['avg_ssim']:.4f}")
        print(f"  平均 IoU: {stats['avg_iou']:.4f}")
        print(f"  平均 Dice: {stats['avg_dice']:.4f}")
        print(f"  平均 HD95: {stats['avg_hd95']:.4f}")
        
        return stats
    
    high_stats = compute_group_stats(high_confidence, "高置信度组")
    low_stats = compute_group_stats(low_confidence, "低置信度组")
    middle_stats = compute_group_stats(middle_confidence, "中等置信度组")
    
    # 保存详细结果
    results_csv = output_root / "uncertainty_results_per_image.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["filename", "u_pixel", "psnr", "ssim", "iou", "dice", "hd95", "group"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            if result["u_pixel"] < low_threshold:
                group = "high"
            elif result["u_pixel"] > high_threshold:
                group = "low"
            else:
                group = "middle"
            writer.writerow({**result, "group": group})
    
    # 保存分组统计摘要
    summary_txt = output_root / "uncertainty_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("不确定性量化评估摘要\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"模型：{cfg['model']['checkpoint_path']}\n")
        handle.write(f"数据集：{cfg['data'].get('h5_path', 'N/A')}\n")
        handle.write(f"采样次数：{n_samples}\n")
        handle.write(f"高置信度阈值：U_pixel < {low_threshold}\n")
        handle.write(f"低置信度阈值：U_pixel > {high_threshold}\n")
        handle.write("\n")
        
        handle.write(f"样本分组统计：\n")
        handle.write(f"  - 高置信度组：{high_stats['count']} 个样本\n")
        handle.write(f"  - 低置信度组：{low_stats['count']} 个样本\n")
        handle.write(f"  - 中等置信度组：{middle_stats['count']} 个样本\n")
        handle.write(f"  - 总样本数：{len(all_results)}\n")
        handle.write("\n")
        
        handle.write("高置信度组性能：\n")
        handle.write(f"  平均 U_pixel: {high_stats['avg_u_pixel']:.6f}\n")
        handle.write(f"  平均 PSNR: {high_stats['avg_psnr']:.4f}\n")
        handle.write(f"  平均 SSIM: {high_stats['avg_ssim']:.4f}\n")
        handle.write(f"  平均 IoU: {high_stats['avg_iou']:.4f}\n")
        handle.write(f"  平均 Dice: {high_stats['avg_dice']:.4f}\n")
        handle.write(f"  平均 HD95: {high_stats['avg_hd95']:.4f}\n")
        handle.write("\n")
        
        handle.write("低置信度组性能：\n")
        handle.write(f"  平均 U_pixel: {low_stats['avg_u_pixel']:.6f}\n")
        handle.write(f"  平均 PSNR: {low_stats['avg_psnr']:.4f}\n")
        handle.write(f"  平均 SSIM: {low_stats['avg_ssim']:.4f}\n")
        handle.write(f"  平均 IoU: {low_stats['avg_iou']:.4f}\n")
        handle.write(f"  平均 Dice: {low_stats['avg_dice']:.4f}\n")
        handle.write(f"  平均 HD95: {low_stats['avg_hd95']:.4f}\n")
        handle.write("\n")
        
        handle.write("中等置信度组性能：\n")
        handle.write(f"  平均 U_pixel: {middle_stats['avg_u_pixel']:.6f}\n")
        handle.write(f"  平均 PSNR: {middle_stats['avg_psnr']:.4f}\n")
        handle.write(f"  平均 SSIM: {middle_stats['avg_ssim']:.4f}\n")
        handle.write(f"  平均 IoU: {middle_stats['avg_iou']:.4f}\n")
        handle.write(f"  平均 Dice: {middle_stats['avg_dice']:.4f}\n")
        handle.write(f"  平均 HD95: {middle_stats['avg_hd95']:.4f}\n")
    
    # 保存分组统计CSV
    group_stats_csv = output_root / "uncertainty_group_stats.csv"
    with group_stats_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["group", "count", "avg_u_pixel", "avg_psnr", "avg_ssim", "avg_iou", "avg_dice", "avg_hd95"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        writer.writerow({"group": "high_confidence", **high_stats})
        writer.writerow({"group": "low_confidence", **low_stats})
        writer.writerow({"group": "middle_confidence", **middle_stats})
    
    print(f"\n结果已保存到 {output_root}")
    print(f"  - 详细结果：{results_csv}")
    print(f"  - 分组统计：{group_stats_csv}")
    print(f"  - 摘要报告：{summary_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="I2SB 不确定性量化评估")
    parser.add_argument("--config",
                        type=Path,
                        default=DEFAULT_CONFIG_PATH,
                        help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device"))
    print(f"使用设备：{device}")
    
    single_stage = env_flag("SR_SINGLE_STAGE")
    if single_stage:
        print("单阶段模式：I2SB仅基于LR条件")
    else:
        print("双阶段模式：UNet粗重建 + I2SB精细化")
    
    evaluate_with_uncertainty(cfg, device, two_stage=not single_stage)


if __name__ == "__main__":
    main()
