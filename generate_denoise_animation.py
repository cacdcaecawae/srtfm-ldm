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

# 添加当前目录到路径以导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

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
        description="Generate UNet+I2SB reverse-denoising animation frames and GIF.")
    parser.add_argument("--config",
                        type=Path,
                        default=Path("I2sb/eval.json"),
                        help="Path to I2SB evaluation configuration JSON file.")
    parser.add_argument("--checkpoint",
                        type=Path,
                        help="Path to I2SB model checkpoint file (overrides config).")
    parser.add_argument("--unet-checkpoint",
                        type=Path,
                        help="Path to UNet model checkpoint file (overrides config).")
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
                        default=999,
                        help="Number of sampling steps (default: 999).")
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
    """构建 UNet 和 I2SB 扩散模型"""
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
    # UNet 只接收单通道输入（intensity），与训练时一致
    unet = build_network(unet_cfg,
                         in_channels=in_channels,
                         image_size=image_size,
                         lr_channels=1,
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
    """
    加载 LR 和 HR 张量对，严格按照 i2sb_eval.py 的流程
    
    返回：
        lr_tensor: LR 输入（3通道 TFM）
        hr_tensor: HR 目标
        sample_name: 样本名称
    """
    data_cfg = cfg["data"]

    if not data_cfg.get("h5_path"):
        raise ValueError("I2SB animation generation requires HDF5 dataset with paired LR/HR images.")

    # 构建 coord_range 参数（与 i2sb_eval.py 一致）
    coord_range = None
    if data_cfg.get("use_tfm_channels", False):
        coord_range_x = tuple(data_cfg["coord_range_x"]) if "coord_range_x" in data_cfg else (-1.0, 1.0)
        coord_range_y = tuple(data_cfg["coord_range_y"]) if "coord_range_y" in data_cfg else (-1.0, 1.0)
        coord_range = (coord_range_x, coord_range_y)
    
    # 加载数据（与 i2sb_eval.py 的 create_dataloader 一致）
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
    生成 I2SB 逆向采样的中间帧，严格复制 diffusion.ddpm_sampling 逻辑
    
    Args:
        diffusion: I2SB 扩散对象
        net: 去噪网络
        condition_tensor: Condition tensor (coarse HR + LR)，即 x1
        num_steps: 采样步数
        ot_ode: 是否使用 OT-ODE
        stride: 帧采样步长
    
    Yields:
        (time_step, image_tensor) 元组
    """
    device = condition_tensor.device
    n_diffusion_steps = diffusion.betas.shape[0]
    
    # 严格按照 i2sb_eval.py 生成 steps
    # step_size = n_diffusion_steps // num_steps
    # steps = list(range(0, n_diffusion_steps, step_size))
    # 等价于 np.linspace(0, n_diffusion_steps-1, num_steps).astype(int) 但用整数步长
    step_size = n_diffusion_steps // num_steps
    steps = list(range(0, n_diffusion_steps, step_size))
    # 确保包含最后一步
    if steps[-1] != n_diffusion_steps - 1:
        steps.append(n_diffusion_steps - 1)
    
    # 严格复制 ddpm_sampling: xt = x1[:,0:1]
    x1 = condition_tensor  # x1 是完整 condition
    xt = x1[:, 0:1].detach().to(device)
    
    # 输出初始状态 (coarse_hr)
    yield steps[-1], xt.detach().cpu()
    
    # 严格复制 ddpm_sampling: steps = steps[::-1], pair_steps = zip(steps[1:], steps[:-1])
    steps_reversed = steps[::-1]
    pair_steps = list(zip(steps_reversed[1:], steps_reversed[:-1]))
    
    with torch.inference_mode():
        for idx, (prev_step, step) in enumerate(tqdm(pair_steps,
                                                     desc='I2SB sampling',
                                                     total=len(pair_steps)),
                                                start=1):
            # 严格复制 ddpm_sampling:
            # pred_x0 = self.pred_x0_fn(xt, step, net, cond=x1)
            pred_x0 = diffusion.pred_x0_fn(xt, step, net, cond=x1)
            
            # xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
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


def save_animation(images: List[Tuple[int, Image.Image]],
                   output_dir: Path,
                   gif_name: str,
                   fps: int,
                   dump_frames: bool,
                   tail_hold: int = 10) -> Path:
    ensure_dir(output_dir)
    gif_path = output_dir / gif_name
    if dump_frames:
        frames_dir = output_dir / (gif_path.stem + "_frames")
        ensure_dir(frames_dir)
        for idx, (t_step, img) in enumerate(images):
            img.convert("RGB").save(frames_dir / f"{idx:04d}_t{t_step:04d}.png")
    duration = max(1, int(1000 / max(1, fps)))
    pil_frames = [img.convert("RGB") for _, img in images]
    if not pil_frames:
        raise RuntimeError("No frames generated for animation.")
    total_frames = pil_frames.copy()
    for _ in range(max(0, tail_hold)):
        total_frames.append(pil_frames[-1])
    pil_frames[0].save(gif_path,
                       save_all=True,
                       append_images=total_frames[1:],
                       duration=duration,
                       loop=0)
    return gif_path


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    if args.device:
        cfg["device"] = args.device
    
    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")
    
    # 获取 sampler 配置（严格按照 i2sb_eval.py）
    sampler_cfg = cfg.get("sampler", {})
    
    # 设置随机种子（优先使用配置文件中的 seed）
    seed = args.seed if args.seed is not None else sampler_cfg.get("seed", 1234)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")
    
    # 获取数据配置
    data_cfg = cfg["data"]
    
    # 构建模型
    unet, net = build_models(cfg, device)
    
    # 加载 I2SB 检查点
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = cfg["model"].get("checkpoint_path")
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
        else:
            raise ValueError("No I2SB checkpoint specified. Use --checkpoint or set model.checkpoint_path in config.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"I2SB checkpoint not found: {checkpoint_path}")

    # 加载 UNet 检查点
    unet_ckpt = args.unet_checkpoint
    if not unet_ckpt:
        unet_ckpt = cfg["model"].get("unet_checkpoint")
        if unet_ckpt:
            unet_ckpt = Path(unet_ckpt)
        else:
            raise ValueError("model.unet_checkpoint is required for UNet+I2SB animation.")
    
    if not unet_ckpt.exists():
        raise FileNotFoundError(f"UNet checkpoint not found: {unet_ckpt}")

    load_checkpoint(net, checkpoint_path, device)
    load_checkpoint(unet, unet_ckpt, device)
    net.eval()
    unet.eval()
    
    # 创建扩散对象
    diffusion = create_diffusion(cfg, device)
    
    # 加载 LR 和 HR 图像（严格按照 i2sb_eval.py 的流程）
    lr_images, hr_tensor, lr_label = load_lr_hr_tensors(cfg, args.lr_image, device)
    print(f"Guidance LR sample: {lr_label}")
    print(f"LR shape: {lr_images.shape}, HR shape: {hr_tensor.shape}")
    
    # 获取参数（严格从配置读取，命令行参数可覆盖）
    # num_steps: 优先命令行，否则从 sampler.num_steps 读取
    num_steps = args.num_steps if args.num_steps != 999 else sampler_cfg.get("num_steps", 100)
    # ot_ode: 优先命令行，否则从 sampler.ot_ode 读取
    ot_ode = args.ot_ode if args.ot_ode else sampler_cfg.get("ot_ode", False)
    stride = adjust_stride(num_steps, args.frame_stride, args.max_frames)
    use_jet = data_cfg.get("channels", 1) == 1
    
    print(f"Generating animation with {num_steps} steps (from config: sampler.num_steps)")
    print(f"OT-ODE: {ot_ode} (from config: sampler.ot_ode)")
    print(f"Frame stride: {stride}")
    
    # 严格按照 i2sb_eval.py 的流程：
    # coarse_hr = unet(lr_images[:, :1])
    # condition = torch.cat([coarse_hr, lr_images], dim=1)
    with torch.inference_mode():
        coarse_hr = unet(lr_images[:, :1])
    print(f"Coarse HR shape: {coarse_hr.shape}")
    
    # 保存 UNet 输出用于调试
    value_range = tuple(data_cfg.get("value_range", [-1.0, 1.0]))
    unet_output = denormalize(coarse_hr[0], value_range)
    unet_img = tensor_to_image(unet_output, apply_jet=use_jet)
    output_dir = args.output_dir
    ensure_dir(output_dir)
    safe_label = lr_label.replace(os.sep, "_").replace("/", "_")
    unet_output_path = output_dir / f"{safe_label}_unet_output.png"
    unet_img.convert("RGB").save(unet_output_path)
    print(f"Saved UNet output to: {unet_output_path}")
    
    # 拼接条件: [coarse_hr, lr_images]（严格与 i2sb_eval.py 一致）
    condition = torch.cat([coarse_hr, lr_images], dim=1)
    print(f"Condition shape: {condition.shape}")

    # 生成 I2SB 帧序列
    frames = iterate_i2sb_frames(
        diffusion,
        net,
        condition,
        num_steps,
        ot_ode,
        stride
    )
    
    frame_sequence = list(frames)
    
    # 从配置读取阈值（与 i2sb_eval.py 一致）
    sampler_cfg = cfg.get("sampler", {})
    threshold = sampler_cfg.get("threshold", 0.9)
    
    images = tensors_to_images(
        frame_sequence,
        sample_index=0,
        use_jet=use_jet,
        value_range=value_range,
        threshold=threshold,
        compress_factor=None,  # 不压缩
        tail_frames=10
    )
    
    # 保存动画
    default_stem = Path(safe_label).stem or "lr_sample"
    mode_suffix = "ot" if ot_ode else "stochastic"
    gif_name = args.gif_name or f"{default_stem}_unet_i2sb_{mode_suffix}.gif"
    
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
    # Example usage:
    # python generate_denoise_animation.py --config I2sb/eval.json --lr-image "933_1"--num-steps 100
    # python generate_denoise_animation.py --config I2sb/eval.json --lr-image 0 --num-steps 100 --ot-ode
    # python generate_denoise_animation.py --config I2sb/eval.json --lr-image "972_2" --ot-ode
