#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化评估数据集中的指定样本

用法:
    python augment/visualize_eval_sample.py --sample 901_1
    python augment/visualize_eval_sample.py --sample 0008 --h5 data/evalexp_augmented.h5
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_sample(h5_path: str, sample_name: str, lr_key: str = "TFM", 
                    hr_key: str = "hr", lr_dataset: str = "intensity",
                    transpose_lr: bool = True, transpose_hr: bool = True) -> None:
    """可视化指定样本的LR和HR"""
    
    with h5py.File(h5_path, "r") as f:
        # 检查样本是否存在
        if lr_key not in f or hr_key not in f:
            print(f"错误: HDF5文件缺少 '{lr_key}' 或 '{hr_key}' 分组")
            return
        
        lr_group = f[lr_key]
        hr_group = f[hr_key]
        
        if sample_name not in lr_group or sample_name not in hr_group:
            available = sorted(set(lr_group.keys()) & set(hr_group.keys()))
            print(f"错误: 样本 '{sample_name}' 不存在")
            print(f"可用样本: {available[:10]}..." if len(available) > 10 else f"可用样本: {available}")
            return
        
        # 读取数据
        lr_node = lr_group[sample_name]
        hr_node = hr_group[sample_name]
        
        # 读取强度
        if isinstance(lr_node, h5py.Dataset):
            intensity = lr_node[:]
        else:
            intensity = lr_node[lr_dataset][:]
        
        # 读取HR
        if isinstance(hr_node, h5py.Dataset):
            hr = hr_node[:]
        else:
            hr = hr_node["data"][:]
        
        # 尝试读取X, Y坐标（尝试大写和小写）
        x_coord, y_coord = None, None
        if isinstance(lr_node, h5py.Group):
            if "X" in lr_node:
                x_coord = lr_node["X"][:]
            elif "x" in lr_node:
                x_coord = lr_node["x"][:]
            
            if "Y" in lr_node:
                y_coord = lr_node["Y"][:]
            elif "y" in lr_node:
                y_coord = lr_node["y"][:]
        
        # MATLAB转置处理（MATLAB列优先 → Python行优先）
        if transpose_lr:
            intensity = intensity.T
            if x_coord is not None:
                x_coord = x_coord.T
            if y_coord is not None:
                y_coord = y_coord.T
        
        if transpose_hr:
            hr = hr.T
        
        # 打印信息
        print(f"\n样本: {sample_name}")
        print(f"LR 强度形状: {intensity.shape}, 范围: [{intensity.min():.4f}, {intensity.max():.4f}]")
        print(f"HR 形状: {hr.shape}, 范围: [{hr.min():.4f}, {hr.max():.4f}]")
        if x_coord is not None:
            print(f"X 坐标形状: {x_coord.shape}, 范围: [{x_coord.min():.6f}, {x_coord.max():.6f}]")
        if y_coord is not None:
            print(f"Y 坐标形状: {y_coord.shape}, 范围: [{y_coord.min():.6f}, {y_coord.max():.6f}]")
        
        # 可视化
        num_plots = 2 + (2 if x_coord is not None and y_coord is not None else 0)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        if num_plots == 1:
            axes = [axes]
        
        idx = 0
        
        # LR强度
        im0 = axes[idx].imshow(intensity, cmap='jet')
        axes[idx].set_title(f'LR Intensity\n{intensity.shape}')
        axes[idx].axis('off')
        plt.colorbar(im0, ax=axes[idx])
        idx += 1
        
        # X坐标
        if x_coord is not None:
            im1 = axes[idx].imshow(x_coord, cmap='viridis')
            axes[idx].set_title(f'X Coordinate\n{x_coord.shape}')
            axes[idx].axis('off')
            plt.colorbar(im1, ax=axes[idx])
            idx += 1
        
        # Y坐标
        if y_coord is not None:
            im2 = axes[idx].imshow(y_coord, cmap='viridis')
            axes[idx].set_title(f'Y Coordinate\n{y_coord.shape}')
            axes[idx].axis('off')
            plt.colorbar(im2, ax=axes[idx])
            idx += 1
        
        # HR（自动检测是否为二值图像）
        unique_hr_vals = np.unique(hr)
        is_binary = len(unique_hr_vals) <= 2
        
        if is_binary:
            # 二值图像：使用灰度colormap，关闭插值
            im3 = axes[idx].imshow(hr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[idx].set_title(f'HR Ground Truth (Binary)\n{hr.shape}')
        else:
            # 连续图像：使用jet colormap
            im3 = axes[idx].imshow(hr, cmap='jet')
            axes[idx].set_title(f'HR Ground Truth\n{hr.shape}')
        
        axes[idx].axis('off')
        plt.colorbar(im3, ax=axes[idx])
        
        plt.suptitle(f'Sample: {sample_name}', fontsize=14, y=0.98)
        plt.tight_layout()
        
        # 保存
        output_path = Path(f"./augment/visualize_{sample_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 可视化已保存至: {output_path}")
        
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="可视化评估数据集样本")
    parser.add_argument("--sample", type=str, required=True, help="样本名称，如 301_2")
    parser.add_argument("--h5", type=str, default="./data/output_merge/eval.h5", 
                       help="HDF5文件路径")
    parser.add_argument("--lr_key", type=str, default="TFM", help="LR分组名称")
    parser.add_argument("--hr_key", type=str, default="hr", help="HR分组名称")
    parser.add_argument("--lr_dataset", type=str, default="intensity", 
                       help="LR dataset名称")
    parser.add_argument("--no_transpose_lr", action="store_true",
                       help="不转置LR数据")
    parser.add_argument("--no_transpose_hr", action="store_true",
                       help="不转置HR数据")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_sample(args.h5, args.sample, args.lr_key, args.hr_key, args.lr_dataset,
                    transpose_lr=not args.no_transpose_lr,
                    transpose_hr=not args.no_transpose_hr)


if __name__ == "__main__":
    main()
