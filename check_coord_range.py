"""检查 HDF5 数据集中所有样本的 X/Y 坐标范围"""
import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def check_coord_ranges(h5_path: str, group_name: str = "TFM") -> dict:
    """
    检查 HDF5 文件中所有样本的坐标范围
    
    Args:
        h5_path: HDF5 文件路径
        group_name: 数据组名称（默认 "TFM"）
    
    Returns:
        包含统计信息的字典
    """
    print(f"正在分析文件: {h5_path}")
    print(f"数据组: {group_name}\n")
    
    with h5py.File(h5_path, 'r') as f:
        if group_name not in f:
            raise KeyError(f"文件中不存在组 '{group_name}'")
        
        group = f[group_name]
        sample_names = sorted(group.keys())
        
        print(f"总样本数: {len(sample_names)}\n")
        
        # 存储所有样本的坐标范围
        all_x_mins, all_x_maxs = [], []
        all_y_mins, all_y_maxs = [], []
        
        # 检查第一个样本的结构
        first_sample = group[sample_names[0]]
        has_x = 'X' in first_sample or 'x' in first_sample
        has_y = 'Y' in first_sample or 'y' in first_sample
        
        if not has_x or not has_y:
            print("❌ 样本中没有找到 X/Y 坐标数据")
            print(f"第一个样本的键: {list(first_sample.keys())}")
            return {}
        
        x_key = 'X' if 'X' in first_sample else 'x'
        y_key = 'Y' if 'Y' in first_sample else 'y'
        
        print(f"坐标键名: X -> '{x_key}', Y -> '{y_key}'\n")
        print("正在扫描所有样本...")
        
        # 遍历所有样本
        for sample_name in tqdm(sample_names, desc="处理样本"):
            sample = group[sample_name]
            
            X = sample[x_key][:]
            Y = sample[y_key][:]
            
            all_x_mins.append(X.min())
            all_x_maxs.append(X.max())
            all_y_mins.append(Y.min())
            all_y_maxs.append(Y.max())
        
        # 统计结果
        x_min_global = float(np.min(all_x_mins))
        x_max_global = float(np.max(all_x_maxs))
        y_min_global = float(np.min(all_y_mins))
        y_max_global = float(np.max(all_y_maxs))
        
        x_min_mean = float(np.mean(all_x_mins))
        x_max_mean = float(np.mean(all_x_maxs))
        y_min_mean = float(np.mean(all_y_mins))
        y_max_mean = float(np.mean(all_y_maxs))
        
        x_min_std = float(np.std(all_x_mins))
        x_max_std = float(np.std(all_x_maxs))
        y_min_std = float(np.std(all_y_mins))
        y_max_std = float(np.std(all_y_maxs))
        
        # 打印结果
        print("\n" + "="*70)
        print("                    坐标范围统计结果")
        print("="*70)
        
        print(f"\n【X 坐标】")
        print(f"  全局范围:  [{x_min_global:.6f}, {x_max_global:.6f}]")
        print(f"  平均范围:  [{x_min_mean:.6f}, {x_max_mean:.6f}]")
        print(f"  标准差:     min={x_min_std:.6f}, max={x_max_std:.6f}")
        print(f"  跨度:       {x_max_global - x_min_global:.6f}")
        
        print(f"\n【Y 坐标】")
        print(f"  全局范围:  [{y_min_global:.6f}, {y_max_global:.6f}]")
        print(f"  平均范围:  [{y_min_mean:.6f}, {y_max_mean:.6f}]")
        print(f"  标准差:     min={y_min_std:.6f}, max={y_max_std:.6f}")
        print(f"  跨度:       {y_max_global - y_min_global:.6f}")
        
        print("\n" + "="*70)
        print("                建议的配置参数")
        print("="*70)
        
        # 建议的范围（留10%余量）
        x_margin = (x_max_global - x_min_global) * 0.1
        y_margin = (y_max_global - y_min_global) * 0.1
        
        x_range_min = x_min_global - x_margin
        x_range_max = x_max_global + x_margin
        y_range_min = y_min_global - y_margin
        y_range_max = y_max_global + y_margin
        
        print(f'\n"coord_range_x": [{x_range_min:.6f}, {x_range_max:.6f}],')
        print(f'"coord_range_y": [{y_range_min:.6f}, {y_range_max:.6f}],')
        
        print("\n或者使用精确范围（无余量）：")
        print(f'"coord_range_x": [{x_min_global:.6f}, {x_max_global:.6f}],')
        print(f'"coord_range_y": [{y_min_global:.6f}, {y_max_global:.6f}],')
        
        print("\n" + "="*70)
        
        # 检查坐标是否一致
        if x_min_std < 1e-6 and x_max_std < 1e-6 and y_min_std < 1e-6 and y_max_std < 1e-6:
            print("\n✓ 所有样本的坐标范围完全一致")
            print("  → 建议: 可以考虑不使用坐标通道 (use_tfm_channels: false)")
        else:
            print("\n⚠ 不同样本的坐标范围有差异")
            print("  → 建议: 需要使用坐标通道 (use_tfm_channels: true)")
        
        print("="*70 + "\n")
        
        return {
            'x_min': x_min_global,
            'x_max': x_max_global,
            'y_min': y_min_global,
            'y_max': y_max_global,
            'x_consistent': x_min_std < 1e-6 and x_max_std < 1e-6,
            'y_consistent': y_min_std < 1e-6 and y_max_std < 1e-6,
        }


def main():
    parser = argparse.ArgumentParser(description="检查 HDF5 数据集的坐标范围")
    parser.add_argument('--train', type=str, default='./data/output_merge/train.h5',
                        help='训练集 HDF5 文件路径')
    parser.add_argument('--eval', type=str, default='./data/output_merge/eval.h5',
                        help='评估集 HDF5 文件路径')
    parser.add_argument('--group', type=str, default='TFM',
                        help='数据组名称')
    parser.add_argument('--skip-eval', action='store_true',
                        help='跳过评估集检查')
    
    args = parser.parse_args()
    
    # 检查训练集
    if Path(args.train).exists():
        train_stats = check_coord_ranges(args.train, args.group)
    else:
        print(f"❌ 训练集文件不存在: {args.train}\n")
    
    # 检查评估集
    if not args.skip_eval and Path(args.eval).exists():
        print("\n" + "#"*70 + "\n")
        eval_stats = check_coord_ranges(args.eval, args.group)
    elif not args.skip_eval:
        print(f"⚠ 评估集文件不存在: {args.eval}")


if __name__ == "__main__":
    main()
