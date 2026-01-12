import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径以导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import H5PairedDataset
from I2sb.diffusion import Diffusion
from network.networksb import (build_network, convnet_big_cfg, convnet_medium_cfg,
                     convnet_small_cfg, unet_res_cfg,
                     unet_res_diffusion_cfg)


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "convnet_small": convnet_small_cfg,
    "convnet_medium": convnet_medium_cfg,
    "convnet_big": convnet_big_cfg,
    "unet_res": unet_res_cfg,
    "unet_res_diffusion": unet_res_diffusion_cfg,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate I2SB reverse-denoising animation frames and GIF.")
    parser.add_argument("--config",
                        type=Path,
                        default=Path("I2sb/eval.json"),
                        help="Path to I2SB evaluation configuration JSON file.")
    parser.add_argument("--checkpoint",
                        type=Path,
                        help="Path to model checkpoint file (overrides config).")
    parser.add_argument("--lr-image",
                        type=str,
                        help="Guidance sample identifier (index or name in HDF5).")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("SR/animations"),
                        help="Directory to place generated artifacts.")
    parser.add_argument("--gif-name",
                        type=str,
                        help="Optional GIF filename (defaults to lr image stem).")
    parser.add_argument("--fps",
                        type=int,
                        default=10,
                        help="Playback rate for the exported GIF.")
    parser.add_argument("--frame-stride",
                        type=int,
                        default=1,
                        help="Keep one frame every N diffusion steps.")
    parser.add_argument("--max-frames",
                        type=int,
                        help="Optional ceiling on saved frames (stride auto adjusts).")
    parser.add_argument("--seed",
                        type=int,
                        help="Random seed for noise sampling.")
    parser.add_argument("--skip-frame-dump",
                        action="store_true",
                        help="Only export GIF, skip individual frame PNGs.")
    parser.add_argument("--device",
                        type=str,
                        help="Override device string from config (e.g. cpu or cuda:0).")
    parser.add_argument("--ot-ode",
                        action="store_true",
                        help="Use OT-ODE (deterministic) sampling instead of stochastic.")
    parser.add_argument("--num-steps",
                        type=int,
                        default=999
                        ,
                        help="Number of sampling steps (default: 100).")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None:
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


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
                 device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    i2sb_backbone_key = model_cfg.get("i2sb_backbone",
                                      model_cfg.get("backbone", "unet_res_diffusion"))
    if i2sb_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown backbone '{i2sb_backbone_key}'. "
                       f"Available: {', '.join(MODEL_CONFIGS)}")
    unet_backbone_key = model_cfg.get("unet_backbone", "unet_res")
    if unet_backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown UNet backbone '{unet_backbone_key}'. "
                       f"Available: {', '.join(MODEL_CONFIGS)}")
    i2sb_cfg = MODEL_CONFIGS[i2sb_backbone_key].copy()
    unet_cfg = MODEL_CONFIGS[unet_backbone_key].copy()

    n_steps = model_cfg["diffusion_steps"]
    t0 = float(model_cfg.get("t0", 1e-4))
    T = float(model_cfg.get("T", 1.0))
    noise_levels = torch.linspace(t0, T, n_steps, device=device, dtype=torch.float32) * n_steps

    in_channels = data_cfg["channels"]
    image_size = data_cfg["image_size"]

    lr_channels = 3 if data_cfg.get("use_tfm_channels", False) else in_channels
    i2sb_lr_channels = in_channels + lr_channels

    i2sb_net = build_network(i2sb_cfg,
                             in_channels=in_channels,
                             image_size=image_size,
                             lr_channels=i2sb_lr_channels,
                             n_steps=n_steps,
                             noise_levels=noise_levels).to(device)
    unet = build_network(unet_cfg,
                         in_channels=in_channels,
                         image_size=image_size,
                         lr_channels=lr_channels,
                         n_steps=None).to(device)
    return unet, i2sb_net


def load_checkpoint(net: torch.nn.Module, checkpoint_path: Path,
                    device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = select_state_dict(checkpoint)
    net.load_state_dict(state_dict)
    print(f"Loaded weights from {checkpoint_path}")


def resolve_h5_index(dataset: H5PairedDataset, target: Optional[str]) -> int:
    if target is None:
        return 0
    try:
        index = int(target)
        if 0 <= index < len(dataset):
            return index
    except ValueError:
        pass
    for index in range(len(dataset)):
        _, _, name = dataset[index]
        if name == target:
            return index
    raise ValueError(f"Could not locate sample '{target}' in {dataset.h5_path}.")


def load_lr_hr_tensors(cfg: Dict[str, Any],
                       lr_name: Optional[str],
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """加载 LR 和 HR 张量对，用于 I2SB 采样"""
    data_cfg = cfg["data"]
    image_size = data_cfg["image_size"]
    channels = data_cfg["channels"]

    if not data_cfg.get("h5_path"):
        raise ValueError("I2SB animation generation requires HDF5 dataset with paired LR/HR images.")

    # 构建 coord_range 参数
    coord_range = None
    if data_cfg.get("use_tfm_channels", False):
        coord_range_x = tuple(data_cfg["coord_range_x"]) if "coord_range_x" in data_cfg else (-1.0, 1.0)
        coord_range_y = tuple(data_cfg["coord_range_y"]) if "coord_range_y" in data_cfg else (-1.0, 1.0)
        coord_range = (coord_range_x, coord_range_y)
    
    dataset = H5PairedDataset(
        h5_path=data_cfg["h5_path"],
        lr_key=data_cfg.get("h5_lr_key", "TFM"),
        hr_key=data_cfg.get("h5_hr_key", "hr"),
        lr_dataset_name=data_cfg.get("h5_lr_dataset"),
        hr_dataset_name=data_cfg.get("h5_hr_dataset"),
        transpose_lr=data_cfg.get("transpose_lr", False),
        transpose_hr=data_cfg.get("transpose_hr", False),
        use_tfm_channels=data_cfg.get("use_tfm_channels", False),
        coord_range=coord_range,
        augment=False,
    )
    
    index = resolve_h5_index(dataset, lr_name)
    lr_tensor, hr_tensor, sample_name = dataset[index]
    
    # 添加batch维度并移到设备
    lr_tensor = lr_tensor.unsqueeze(0).to(device)
    hr_tensor = hr_tensor.unsqueeze(0).to(device)
    
    return lr_tensor, hr_tensor, sample_name


def make_beta_schedule(n_timestep: int = 1000,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2) -> np.ndarray:
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
                      n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def create_diffusion(cfg: Dict[str, Any], device: torch.device) -> Diffusion:
    """创建 I2SB 扩散对象"""
    n_steps = cfg["model"]["diffusion_steps"]
    # 使用线性 beta schedule
    betas = make_beta_schedule(n_timestep=n_steps, linear_end=3e-4)
    half = n_steps // 2
    if n_steps % 2 == 1:
        betas = np.concatenate([betas[:half], [betas[half]], np.flip(betas[:half])])
    else:
        betas = np.concatenate([betas[:half], np.flip(betas[:half])])
    diffusion = Diffusion(betas, device)
    return diffusion


def adjust_stride(total_steps: int,
                  stride: int,
                  max_frames: Optional[int]) -> int:
    stride = max(1, int(stride))
    if not max_frames:
        return stride
    max_frames = max(1, int(max_frames))
    stride = max(stride, (total_steps + max_frames - 1) // max_frames)
    return stride


def iterate_i2sb_frames(
        diffusion: Diffusion,
        net: torch.nn.Module,
        condition_tensor: torch.Tensor,
        num_steps: int,
        ot_ode: bool,
        stride: int) -> Iterable[Tuple[int, torch.Tensor]]:
    """
    生成 I2SB 逆向采样的中间帧
    
    参考 I2sb/diffusion.py 中的 ddpm_sampling 实现
    
    Args:
        diffusion: I2SB 扩散对象
        net: 去噪网络
        condition_tensor: Condition tensor (x1, coarse HR + LR)
        num_steps: 采样步数
        ot_ode: 是否使用 OT-ODE
        stride: 帧采样步长
    
    Yields:
        (time_step, image_tensor) 元组
    """
    device = condition_tensor.device
    n_steps = diffusion.betas.shape[0]
    
    # 生成时间步序列 [0, step_size, 2*step_size, ..., n_steps]
    step_size = n_steps // num_steps
    steps = np.arange(0, n_steps + 1, step_size)
    if steps[-1] != n_steps:
        steps = np.append(steps, n_steps)
    
    # x1 uses coarse HR in the first channel, matching ddpm_sampling.
    xt = condition_tensor[:, 0:1].detach().to(device)
    
    # 输出初始状态
    yield n_steps, xt.detach().cpu()
    
    # 反向遍历时间步
    steps_reversed = steps[::-1]
    pair_steps = list(zip(steps_reversed[1:], steps_reversed[:-1]))
    
    with torch.inference_mode():
        for idx, (prev_step, step) in enumerate(tqdm(pair_steps,
                                                     desc='I2SB sampling',
                                                     total=len(pair_steps)),
                                                start=1):
            # 预测 x0 (参考 pred_x0_fn 的实现)
            # 确保 prev_step 和 step 都在有效范围内，且 prev_step < step
            prev_step = min(prev_step, n_steps - 1)
            step = min(step, n_steps - 1)
            if prev_step >= step:
                continue  # 跳过无效的时间步对
            step_tensor = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
            out = net(xt, step_tensor, condition_tensor)
            
            # 计算 pred_x0 (参考 compute_pred_x0)
            std_fwd = diffusion.std_fwd[step]
            if xt.dim() > 1:
                # unsqueeze_xdim
                std_fwd = std_fwd.view(-1, *([1] * (xt.dim() - 1)))
            pred_x0 = xt - std_fwd * out
            
            # 通过后验分布采样 x_{prev_step}
            xt = diffusion.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            
            # 根据步长输出中间结果
            if idx % stride == 0 or prev_step == 0:
                yield prev_step, xt.detach().cpu()


def denormalize(tensor: torch.Tensor, value_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """将张量从 value_range 反归一化到 [0, 1]"""
    vmin, vmax = value_range
    return (tensor - vmin) / (vmax - vmin)


def tensor_to_image(tensor: torch.Tensor, apply_jet: bool = False) -> Image.Image:
    """将张量转换为 PIL Image"""
    tensor = tensor.clamp(0.0, 1.0)
    
    if apply_jet:
        # 单通道图像使用 jet colormap
        array = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        from matplotlib import cm
        try:
            from matplotlib import colormaps
            cmap = colormaps.get_cmap('jet')
        except (ImportError, AttributeError):
            cmap = cm.get_cmap('jet')
        colored = cmap(array)
        colored = (colored[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored, mode='RGB')
    else:
        # 多通道或RGB图像
        if tensor.shape[0] == 1:
            array = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(array, mode='L')
        elif tensor.shape[0] == 3:
            array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(array, mode='RGB')
        else:
            # 多通道情况，只显示第一个通道
            array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(array, mode='L')


def tensors_to_images(frames: Iterable[Tuple[int, torch.Tensor]],
                      sample_index: int,
                      use_jet: bool,
                      value_range: Tuple[float, float] = (-1.0, 1.0),
                      threshold: Optional[float] = None,
                      compress_factor: Optional[float] = None,
                      tail_frames: int = 0) -> List[Tuple[int, Image.Image]]:
    """
    将张量帧序列转换为图像列表
    
    Args:
        frames: (time_step, tensor) 迭代器
        sample_index: batch 中的样本索引
        use_jet: 是否使用 jet colormap
        value_range: 张量值域
        threshold: 阈值处理参数
        compress_factor: 压缩因子
        tail_frames: 尾部渐变帧数
    """
    frame_list = list(frames)
    total = len(frame_list)
    images: List[Tuple[int, Image.Image]] = []
    
    for idx, (t_step, tensor) in enumerate(frame_list):
        if sample_index >= tensor.shape[0]:
            raise IndexError(
                f"Sample index {sample_index} out of range for tensor batch "
                f"of size {tensor.shape[0]}.")
        
        # 反归一化
        original = denormalize(tensor[sample_index], value_range)
        frame_tensor = original
        
        # 处理尾部渐变效果
        if tail_frames > 0 and idx >= total - tail_frames:
            processed = original
            if threshold is not None:
                processed = torch.where(
                    processed < threshold,
                    torch.zeros_like(processed),
                    processed)
            if compress_factor is not None:
                processed = processed * compress_factor
            processed = processed.clamp(0.0, 1.0)
            
            if tail_frames == 1:
                blend = 1.0
            else:
                blend = float(idx - (total - tail_frames)) / float(tail_frames - 1)
            frame_tensor = torch.lerp(original, processed, blend)
        
        images.append((t_step, tensor_to_image(frame_tensor, apply_jet=use_jet)))
    
    return images


def save_animation(images: List[Tuple[int, Image.Image]],
                   output_dir: Path,
                   gif_name: str,
                   fps: int,
                   dump_frames: bool,
                   tail_hold: int = 10) -> Path:
    """保存动画为 GIF 文件"""
    ensure_dir(output_dir)
    gif_path = output_dir / gif_name
    
    # 保存单独的帧
    if dump_frames:
        frames_dir = output_dir / (gif_path.stem + "_frames")
        ensure_dir(frames_dir)
        for idx, (t_step, img) in enumerate(images):
            img.convert("RGB").save(frames_dir / f"{idx:04d}_t{t_step:04d}.png")
    
    # 创建 GIF
    duration = max(1, int(1000 / max(1, fps)))
    pil_frames = [img.convert("RGB") for _, img in images]
    
    if not pil_frames:
        raise RuntimeError("No frames generated for animation.")
    
    # 添加尾部保持帧
    total_frames = pil_frames.copy()
    for _ in range(max(0, tail_hold)):
        total_frames.append(pil_frames[-1])
    
    pil_frames[0].save(gif_path,
                       save_all=True,
                       append_images=total_frames[1:],
                       duration=duration,
                       loop=0)
    return gif_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    if args.device:
        cfg["device"] = args.device
    
    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    else:
        print("Warning: No random seed specified, results may not be reproducible.")
    
    # 获取数据配置
    data_cfg = cfg["data"]
    
    # 构建模型
    unet, net = build_models(cfg, device)
    
    # Load checkpoints
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = Path(cfg["model"].get("checkpoint_path"))
        if not checkpoint_path:
            raise ValueError("No checkpoint specified. Use --checkpoint or set model.checkpoint_path in config.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    unet_ckpt = cfg["model"].get("unet_checkpoint")
    if not unet_ckpt:
        raise ValueError("model.unet_checkpoint is required for two-stage I2SB sampling.")
    unet_ckpt = Path(unet_ckpt)
    if not unet_ckpt.exists():
        raise FileNotFoundError(f"UNet checkpoint not found: {unet_ckpt}")

    load_checkpoint(net, checkpoint_path, device)
    load_checkpoint(unet, unet_ckpt, device)
    net.eval()
    unet.eval()
    
    # 创建扩散对象
    diffusion = create_diffusion(cfg, device)
    
    # 加载 LR 和 HR 图像
    lr_tensor, hr_tensor, lr_label = load_lr_hr_tensors(cfg, args.lr_image, device)
    print(f"Guidance LR sample: {lr_label}")
    print(f"LR shape: {lr_tensor.shape}, HR shape: {hr_tensor.shape}")
    
    # 获取参数
    num_steps = args.num_steps
    ot_ode = args.ot_ode
    stride = adjust_stride(num_steps, args.frame_stride, args.max_frames)
    use_jet = data_cfg.get("channels", 1) == 1
    
    print(f"Generating animation with {num_steps} steps")
    print(f"OT-ODE: {ot_ode}")
    print(f"Frame stride: {stride}")
    
    # Generate frame sequence from the coarse+LR condition.
    with torch.inference_mode():
        coarse_hr = unet(lr_tensor)
    condition = torch.cat([coarse_hr, lr_tensor], dim=1)

    frames = iterate_i2sb_frames(
        diffusion,
        net,
        condition,
        num_steps,
        ot_ode,
        stride
    )
    
    frame_sequence = list(frames)
    
    # 转换为图像
    value_range = tuple(data_cfg.get("value_range", [-1.0, 1.0]))
    threshold = 0.95
    compress_factor = 227.0 / 253.0
    
    images = tensors_to_images(
        frame_sequence,
        sample_index=0,
        use_jet=use_jet,
        value_range=value_range,
        threshold=threshold,
        compress_factor=compress_factor,
        tail_frames=10
    )
    
    # 保存动画
    output_dir = args.output_dir
    safe_label = lr_label.replace(os.sep, "_").replace("/", "_")
    default_stem = Path(safe_label).stem or "lr_sample"
    mode_suffix = "ot" if ot_ode else "stochastic"
    gif_name = args.gif_name or f"{default_stem}_i2sb_{mode_suffix}.gif"
    
    gif_path = save_animation(
        images,
        output_dir,
        gif_name,
        fps=args.fps,
        dump_frames=not args.skip_frame_dump,
        tail_hold=10
    )
    
    print(f"Saved animation to: {gif_path}")


if __name__ == "__main__":
    main()
    # Example usage:eval.json --lr-image 0 --num-steps 100
    # python I2sb/generate_i2sb_animation.py --config I2sb/eval.json --lr-image 0 --num-steps 100 --ot-ode
    # python I2sb/generate_i2sb_animation.py --config I2sb/eval.json --checkpoint SR/weight_ckpt/I2sb1124.pth --lr-image 0
    # python I2sb/generate_i2sb_animation.py --config I2sb/train.json --checkpoint SR/weight_ckpt/I2sb1124.pth --lr-image 0 --num-steps 100 --ot-ode
    # python I2sb/generate_i2sb_animation.py --config I2sb/eval.json --lr-image "972_2" --ot-ode
