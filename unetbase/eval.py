import argparse
import csv
import json
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
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_h5_dataloader
from network.network import build_network, unet_res_cfg

JET_PALETTE: List[int] = (
    (cm.jet(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0)
    .astype(np.uint8)
    .reshape(-1)
    .tolist()
)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "unet_res": unet_res_cfg,
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
        augment=False,
        num_workers=cfg.get("num_workers", 4),
        shuffle=False,
    )


def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """构建模型"""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    backbone_key = model_cfg["backbone"]
    
    if backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown backbone '{backbone_key}'. "
                       f"Available: {', '.join(MODEL_CONFIGS)}")
    
    net_cfg = MODEL_CONFIGS[backbone_key].copy()
    
    in_channels = data_cfg["channels"]
    image_size = data_cfg["image_size"]
    
    lr_channels = in_channels
    if data_cfg.get("use_tfm_channels", False):
        lr_channels = 3
    
    model = build_network(net_cfg,
                          in_channels=in_channels,
                          image_size=image_size,
                          lr_channels=lr_channels).to(device)
    
    checkpoint = torch.load(model_cfg["checkpoint_path"], map_location=device)
    state_dict = checkpoint.get("model_best_state_dict",
                                checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_cfg['checkpoint_path']}")
    return model


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


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算Dice系数 (Dice Coefficient)
    
    Args:
        pred: 预测张量 [C, H, W] 或 [B, C, H, W]，值域 [0, 1]
        target: 目标张量，形状与pred相同
        threshold: 二值化阈值
    
    Returns:
        Dice系数值
    """
    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算交集
    intersection = (pred_binary * target_binary).sum()
    
    # 计算Dice系数: 2 * |X ∩ Y| / (|X| + |Y|)
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    # 避免除零
    if pred_sum + target_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + target_sum)
    return dice.item()


def compute_hd95(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, percentile: float = 95.0) -> float:
    """
    计算95th percentile Hausdorff Distance (HD95)
    
    Args:
        pred: 预测张量 [C, H, W] 或 [B, C, H, W]，值域 [0, 1]
        target: 目标张量，形状与pred相同
        threshold: 二值化阈值
        percentile: 百分位数，默认95
    
    Returns:
        HD95值（单位：像素）
    """
    from scipy.ndimage import binary_erosion
    
    # 1. 转为 Numpy 并二值化
    pred_np = (pred > threshold).cpu().numpy().astype(bool)
    target_np = (target > threshold).cpu().numpy().astype(bool)

    # 2. 维度压缩：确保处理的是 [H, W] 的 2D 图像
    # 如果是 [B, C, H, W] 或 [C, H, W]，且我们只关心"是否有缺陷"，则压缩维度
    while pred_np.ndim > 2:
        pred_np = pred_np.any(axis=0)
    while target_np.ndim > 2:
        target_np = target_np.any(axis=0)
    
    # 3. 空值检查
    # 如果两张图都是黑的（都没预测出缺陷，GT也没缺陷），距离为0（完美匹配）
    if not pred_np.any() and not target_np.any():
        return 0.0
    
    # 如果一张有一张没有，返回最大惩罚（对角线距离）
    if not pred_np.any() or not target_np.any():
        h, w = pred_np.shape
        return float(np.sqrt(h**2 + w**2))

    # 4. 计算距离变换
    # distance_transform_edt 计算的是到最近零点的距离
    # 输入 ~pred_np 使得前景点到最近背景点的距离场
    pred_dt = distance_transform_edt(~pred_np) 
    target_dt = distance_transform_edt(~target_np)

    # 5. 提取边界（使用形态学腐蚀 + XOR）
    # 边界 = 原图 XOR 腐蚀后的图
    pred_border = pred_np ^ binary_erosion(pred_np, border_value=0)
    target_border = target_np ^ binary_erosion(target_np, border_value=0)
    
    # 如果边界提取后为空（例如全图都是前景），退化处理
    if not pred_border.any(): 
        pred_border = pred_np
    if not target_border.any(): 
        target_border = target_np

    # 6. 计算距离：
    # Pred 边界上的点，到 Target 最近前景点的距离
    d_pred_to_target = target_dt[pred_border]
    # Target 边界上的点，到 Pred 最近前景点的距离
    d_target_to_pred = pred_dt[target_border]
    
    # 合并所有距离
    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    
    if len(all_distances) == 0:
        return 0.0
        
    return float(np.percentile(all_distances, percentile))


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


def overlay_contour(pred_tensor: torch.Tensor, 
                    gt_tensor: torch.Tensor,
                    apply_jet: bool = False,
                    threshold: float = 0.5,
                    contour_color: tuple = (0, 255, 0)) -> Image.Image:
    """在预测图像上叠加GT轮廓
    
    Args:
        pred_tensor: 预测张量 [C, H, W]，值域 [0, 1]
        gt_tensor: GT张量 [C, H, W]，值域 [0, 1]
        apply_jet: 是否应用jet伪彩色
        threshold: 二值化阈值
        contour_color: 轮廓颜色 (R, G, B)，默认绿色
    
    Returns:
        叠加轮廓后的PIL Image
    """
    # 转换预测图像
    pred_array = pred_tensor.permute(1, 2, 0).cpu().numpy()
    pred_array = np.clip(pred_array, 0.0, 1.0)
    
    # 转换GT为二值掩码
    gt_array = gt_tensor.cpu().numpy()
    if gt_array.ndim == 3:  # [C, H, W]
        gt_array = gt_array.max(axis=0)  # 取最大值投影
    gt_binary = (gt_array > threshold).astype(np.uint8) * 255
    
    # 找到轮廓
    contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果是灰度图或jet伪彩色，先转为RGB
    if apply_jet:
        if pred_array.shape[2] == 1:
            gray_np = pred_array[:, :, 0]
        else:
            gray_np = pred_array.mean(axis=2)
        gray_uint8 = (gray_np * 255.0).astype(np.uint8)
        # 应用jet色图
        img_rgb = cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    else:
        pred_uint8 = (pred_array * 255.0).astype(np.uint8)
        if pred_uint8.shape[2] == 1:
            # 灰度图转RGB
            img_rgb = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = pred_uint8
    
    # 绘制轮廓
    cv2.drawContours(img_rgb, contours, -1, contour_color, thickness=1)
    
    return Image.fromarray(img_rgb)


def evaluate(cfg: Dict[str, Any], device: torch.device) -> None:
    """执行 UNetonly 评估"""
    data_cfg = cfg["data"]
    sampler_cfg = cfg.get("sampler", {})
    
    dataloader = create_dataloader(data_cfg)
    net = build_model(cfg, device)
    
    use_jet = data_cfg.get("channels", 1) == 1
    
    output_root = Path(cfg["output"]["root"])
    images_dir = output_root / "images"
    ensure_dir(images_dir)
    ensure_dir(output_root)
    
    # 推理配置
    seed = sampler_cfg.get("seed", 1234)
    threshold = sampler_cfg.get("threshold", 0.05)
    print("Direct UNet inference")
    
    psnr_scores = []
    ssim_scores = []
    iou_scores = []
    dice_scores = []
    hd95_scores = []
    per_image_results = []
    
    with torch.inference_mode():
        for lr_images, hr_images, names in tqdm(dataloader, desc="Evaluating"):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            sr_images = net(lr_images)
            
            # 反归一化
            sr_for_metric = denormalize(sr_images)
            
            # 阈值处理
            sr_for_metric = torch.where(
                sr_for_metric < threshold,
                torch.zeros_like(sr_for_metric),
                sr_for_metric
            )
            
            # 压缩到 [0, 227/253]
            sr_for_image = sr_for_metric# * (227.0 / 253.0)
            
            hr_for_metric = denormalize(hr_images)
            
            # 计算指标
            for idx, name in enumerate(names):
                pred = sr_for_metric[idx].unsqueeze(0)
                target = hr_for_metric[idx].unsqueeze(0)
                
                psnr = peak_signal_noise_ratio(pred, target, data_range=1.0)
                ssim = structural_similarity_index_measure(pred, target, data_range=1.0)
                iou = compute_iou(pred, target, threshold=threshold)
                dice = compute_dice(pred, target, threshold=threshold)
                hd95 = compute_hd95(pred, target, threshold=threshold)
                
                psnr_scores.append(psnr.item())
                ssim_scores.append(ssim.item())
                iou_scores.append(iou)
                dice_scores.append(dice)
                hd95_scores.append(hd95)
                per_image_results.append({
                    "filename": name,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "iou": iou,
                    "dice": dice,
                    "hd95": hd95,
                })
                
                # 保存图像
                if cfg["output"].get("save_images", True):
                    if not any(name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                        image_filename = f"{name}.png"
                    else:
                        image_filename = name
                    
                    # 保存带轮廓的对比图
                    overlay_contour(sr_for_image[idx],
                                    hr_for_metric[idx],
                                    apply_jet=use_jet,
                                    threshold=threshold).save(images_dir / image_filename)
    
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
    
    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    max_dice = float(np.max(dice_scores)) if dice_scores else 0.0
    min_dice = float(np.min(dice_scores)) if dice_scores else 0.0
    
    avg_hd95 = float(np.mean(hd95_scores)) if hd95_scores else 0.0
    max_hd95 = float(np.max(hd95_scores)) if hd95_scores else 0.0
    min_hd95 = float(np.min(hd95_scores)) if hd95_scores else 0.0
    
    print(f"PSNR - 平均: {avg_psnr:.4f}, 最大: {max_psnr:.4f}, 最小: {min_psnr:.4f}")
    print(f"SSIM - 平均: {avg_ssim:.4f}, 最大: {max_ssim:.4f}, 最小: {min_ssim:.4f}")
    print(f"IoU  - 平均: {avg_iou:.4f}, 最大: {max_iou:.4f}, 最小: {min_iou:.4f}")
    print(f"Dice - 平均: {avg_dice:.4f}, 最大: {max_dice:.4f}, 最小: {min_dice:.4f}")
    print(f"HD95 - 平均: {avg_hd95:.4f}, 最大: {max_hd95:.4f}, 最小: {min_hd95:.4f}")
    
    # 保存结果
    results_txt = output_root / "results.txt"
    data_source = cfg["data"].get("h5_path", "N/A")
    with results_txt.open("w", encoding="utf-8") as handle:
        handle.write("UNetonly Evaluation Summary\n")
        handle.write("======================\n")
        handle.write(f"Model: {cfg['model']['checkpoint_path']}\n")
        handle.write(f"Dataset: {data_source}\n")
        handle.write("\n")
        handle.write(f"PSNR - Average: {avg_psnr:.4f}, Max: {max_psnr:.4f}, Min: {min_psnr:.4f}\n")
        handle.write(f"SSIM - Average: {avg_ssim:.4f}, Max: {max_ssim:.4f}, Min: {min_ssim:.4f}\n")
        handle.write(f"IoU  - Average: {avg_iou:.4f}, Max: {max_iou:.4f}, Min: {min_iou:.4f}\n")
        handle.write(f"Dice - Average: {avg_dice:.4f}, Max: {max_dice:.4f}, Min: {min_dice:.4f}\n")
        handle.write(f"HD95 - Average: {avg_hd95:.4f}, Max: {max_hd95:.4f}, Min: {min_hd95:.4f}\n")
    
    results_csv = output_root / "results_per_image.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "psnr", "ssim", "iou", "dice", "hd95"])
        writer.writeheader()
        writer.writerows(per_image_results)
    
    print(f"Results saved to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UNet SR model.")
    parser.add_argument("--config",
                        type=Path,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to UNetonly evaluation configuration JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")
    evaluate(cfg, device)


if __name__ == "__main__":
    main()
