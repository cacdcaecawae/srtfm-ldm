#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强脚本 - 预先生成增强后的评估数据集

功能:
    从原始 eval.h5 读取 TFM/HR 数据，应用数据增强，保存到新的 HDF5 文件
    保持与 MATLAB 生成的原始格式完全兼容

用法:
    python augment_eval_dataset.py --config augment_config.json
    
输出:
    生成 eval_augmented.h5，结构与原始文件完全相同
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_config(path: Path) -> Dict[str, Any]:
    """加载配置文件"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: Optional[int]) -> None:
    """设置全局随机种子以确保可重复性"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")


def apply_h_flip(
    intensity: np.ndarray,
    x_coord: Optional[np.ndarray],
    y_coord: Optional[np.ndarray],
    hr: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    应用水平翻转
    
    Args:
        intensity: (H, W) 或 (W, H) - MATLAB格式的强度
        x_coord: (H, W) 或 (W, H) - X坐标，可能为None
        y_coord: (H, W) 或 (W, H) - Y坐标，可能为None
        hr: (H, W) 或 (W, H) - MATLAB格式的HR
        
    Returns:
        翻转后的 (intensity, x_coord, y_coord, hr)
    """
    # 水平翻转（沿最后一维）
    intensity_flipped = np.flip(intensity, axis=-1).copy()
    hr_flipped = np.flip(hr, axis=-1).copy()
    
    x_coord_flipped = None
    y_coord_flipped = None
    
    if x_coord is not None:
        # 关键：先取反 X 坐标，再进行空间翻转
        x_coord_flipped = -x_coord
        x_coord_flipped = np.flip(x_coord_flipped, axis=-1).copy()
    
    if y_coord is not None:
        # Y坐标只做空间翻转
        y_coord_flipped = np.flip(y_coord, axis=-1).copy()
    
    return intensity_flipped, x_coord_flipped, y_coord_flipped, hr_flipped


def apply_translation(
    intensity: np.ndarray,
    x_coord: Optional[np.ndarray],
    y_coord: Optional[np.ndarray],
    hr: np.ndarray,
    tx: int,
    ty: int,
    physical_size: float = 0.008
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    应用平移变换
    
    Args:
        intensity: (H, W) 或 (W, H) - MATLAB格式
        x_coord: X坐标
        y_coord: Y坐标
        hr: HR图像
        tx: 水平平移（像素，正数向右）
        ty: 垂直平移（像素，正数向下）
        coord_range: ((x_min, x_max), (y_min, y_max))
        physical_size: 图像物理尺寸（米，用于计算坐标偏移量
    Returns:
        平移后的数组
    """
    # 转换为torch tensor以使用grid_sample
    intensity_t = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    hr_t = torch.from_numpy(hr).float().unsqueeze(0).unsqueeze(0)
    
    _, _, H, W = intensity_t.shape
    
    # 构建平移矩阵
    theta = torch.tensor([
        [1, 0, 2.0 * tx / W],
        [0, 1, 2.0 * ty / H]
    ], dtype=torch.float32).unsqueeze(0)
    
    grid = F.affine_grid(theta, [1, 1, H, W], align_corners=False)
    
    # 应用平移（边界复制填充）
    # 所有图像都用nearest保持原始像素值，避免插值引入虚假数据
    intensity_t = F.grid_sample(intensity_t, grid, mode='nearest',
                                padding_mode='border', align_corners=False)
    hr_t = F.grid_sample(hr_t, grid, mode='nearest',
                        padding_mode='border', align_corners=False)
    
    intensity_shifted = intensity_t.squeeze().numpy()
    hr_shifted = hr_t.squeeze().numpy()
    
    x_coord_shifted = None
    y_coord_shifted = None
    
    # 处理坐标通道
    if x_coord is not None and y_coord is not None:
        # 每像素对应的物理坐标偏移量
        dx_coord = tx * (physical_size / (W - 1))
        dy_coord = ty * (physical_size / (H - 1))
        
        # 坐标值直接平移（不做空间移动）
        x_coord_shifted = x_coord + dx_coord
        y_coord_shifted = y_coord + dy_coord
    
    return intensity_shifted, x_coord_shifted, y_coord_shifted, hr_shifted


def augment_sample(
    intensity: np.ndarray,
    x_coord: Optional[np.ndarray],
    y_coord: Optional[np.ndarray],
    hr: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    对单个样本应用数据增强
    
    Args:
        intensity: MATLAB格式的强度数组
        x_coord: X坐标（可能为None）
        y_coord: Y坐标（可能为None）
        hr: MATLAB格式的HR数组
        cfg: 配置字典
        
    Returns:
        增强后的 (intensity, x_coord, y_coord, hr)
    """
    h_flip_prob = cfg.get("h_flip_prob", 0.5)
    translate_prob = cfg.get("translate_prob", 0.5)
    max_translate_ratio = cfg.get("max_translate_ratio", 0.05)
    
    coord_range = None
    if random.random() < h_flip_prob:
        intensity, x_coord, y_coord, hr = apply_h_flip(
            intensity, x_coord, y_coord, hr
        )
    
    # 平移
    if random.random() < translate_prob:
        # 根据MATLAB存储格式确定维度
        shape = intensity.shape
        H, W = shape[-2], shape[-1]
        
        max_tx = int(W * max_translate_ratio)
        max_ty = int(H * max_translate_ratio)
        
        tx = random.randint(-max_tx, max_tx)
        ty = random.randint(-max_ty, max_ty)
        
        if tx != 0 or ty != 0:
            intensity, x_coord, y_coord, hr = apply_translation(
                intensity, x_coord, y_coord, hr,
                tx, ty,
                physical_size=cfg.get("physical_size", 0.008)
            )
    
    return intensity, x_coord, y_coord, hr


def augment_dataset(cfg: Dict[str, Any]) -> None:
    """
    读取原始数据集，应用增强，保存到新文件
    
    Args:
        cfg: 配置字典
    """
    input_path = Path(cfg["input_h5"])
    output_path = Path(cfg["output_h5"])
    
    lr_key = cfg.get("lr_key", "TFM")
    hr_key = cfg.get("hr_key", "hr")
    lr_dataset = cfg.get("lr_dataset", "intensity")
    use_tfm_channels = cfg.get("use_tfm_channels", False)
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"增强配置: h_flip={cfg.get('h_flip_prob', 0.5)}, "
          f"translate={cfg.get('translate_prob', 0.5)}, "
          f"max_ratio={cfg.get('max_translate_ratio', 0.05)}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(input_path, "r") as f_in, \
         h5py.File(output_path, "w") as f_out:
        
        # 获取样本列表
        lr_group_in = f_in[lr_key]
        hr_group_in = f_in[hr_key]
        
        lr_names = set(lr_group_in.keys())
        hr_names = set(hr_group_in.keys())
        shared_names = sorted(lr_names & hr_names)
        
        if not shared_names:
            raise ValueError(f"未找到共同的样本名称")
        
        print(f"找到 {len(shared_names)} 个样本")
        
        # 创建输出分组
        lr_group_out = f_out.create_group(lr_key)
        hr_group_out = f_out.create_group(hr_key)
        
        # 逐样本处理
        for sample_name in tqdm(shared_names, desc="增强样本"):
            # 读取LR数据
            lr_sample_in = lr_group_in[sample_name]
            
            # 读取强度
            if isinstance(lr_sample_in, h5py.Dataset):
                intensity_data = lr_sample_in[:]
            else:
                intensity_data = lr_sample_in[lr_dataset][:]
            
            # 读取坐标（如果使用TFM模式）
            x_coord_data = None
            y_coord_data = None
            
            if use_tfm_channels and isinstance(lr_sample_in, h5py.Group):
                if "X" in lr_sample_in:
                    x_coord_data = lr_sample_in["X"][:]
                elif "x" in lr_sample_in:
                    x_coord_data = lr_sample_in["x"][:]
                
                if "Y" in lr_sample_in:
                    y_coord_data = lr_sample_in["Y"][:]
                elif "y" in lr_sample_in:
                    y_coord_data = lr_sample_in["y"][:]
            
            # 读取HR数据
            hr_sample_in = hr_group_in[sample_name]
            if isinstance(hr_sample_in, h5py.Dataset):
                hr_data = hr_sample_in[:]
            else:
                hr_data = hr_sample_in["data"][:]
            
            # 转置为Python格式 (MATLAB W×H → Python H×W)
            intensity_transposed = intensity_data.T
            x_coord_transposed = x_coord_data.T if x_coord_data is not None else None
            y_coord_transposed = y_coord_data.T if y_coord_data is not None else None
            hr_transposed = hr_data.T
            
            # 应用增强（在转置后的数据上）
            intensity_aug, x_coord_aug, y_coord_aug, hr_aug = augment_sample(
                intensity_transposed, x_coord_transposed, y_coord_transposed, hr_transposed, cfg
            )
            
            # 转置回MATLAB格式 (Python H×W → MATLAB W×H)
            intensity_aug = intensity_aug.T
            x_coord_aug = x_coord_aug.T if x_coord_aug is not None else None
            y_coord_aug = y_coord_aug.T if y_coord_aug is not None else None
            hr_aug = hr_aug.T
            
            # 写入输出文件（保持原始结构）
            # LR分组
            if isinstance(lr_sample_in, h5py.Dataset):
                lr_group_out.create_dataset(sample_name, data=intensity_aug,
                                          dtype=intensity_aug.dtype,
                                          compression='gzip', compression_opts=5)
            else:
                lr_sample_out = lr_group_out.create_group(sample_name)
                lr_sample_out.create_dataset(lr_dataset, data=intensity_aug,
                                           dtype=intensity_aug.dtype,
                                           compression='gzip', compression_opts=5)
                
                if x_coord_aug is not None:
                    lr_sample_out.create_dataset("X", data=x_coord_aug,
                                               dtype=x_coord_aug.dtype,
                                               compression='gzip', compression_opts=5)
                if y_coord_aug is not None:
                    lr_sample_out.create_dataset("Y", data=y_coord_aug,
                                               dtype=y_coord_aug.dtype,
                                               compression='gzip', compression_opts=5)
            
            # HR分组
            if isinstance(hr_sample_in, h5py.Dataset):
                hr_group_out.create_dataset(sample_name, data=hr_aug,
                                          dtype=hr_aug.dtype,
                                          compression='gzip', compression_opts=5)
            else:
                hr_sample_out = hr_group_out.create_group(sample_name)
                hr_sample_out.create_dataset("data", data=hr_aug,
                                           dtype=hr_aug.dtype,
                                           compression='gzip', compression_opts=5)
    
    print(f"✓ 增强数据集已保存至: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成增强后的评估数据集"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("augment/augment_config.json"),
        help="增强配置文件路径"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    # 设置随机种子
    set_seed(cfg.get("seed", 42))
    
    # 执行增强
    augment_dataset(cfg)


if __name__ == "__main__":
    main()
