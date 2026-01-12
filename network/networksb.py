from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== 扩散模型相关类（从 network_d.py 整合）==============

class PositionalEncoding(nn.Module):
    """正弦位置编码，用于时间步嵌入（支持连续浮点时间）。"""

    def __init__(self, max_seq_len: int, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal encoding.")
        self.dim = d_model
        self.half = d_model // 2
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        连续/离散时间步的正弦位置编码（支持浮点时间）。
        
        Args:
            t: 时间步 [B] 或 [B, 1]，可以是整数或浮点数
            
        Returns:
            位置编码 [B, d_model]
        """
        t = t.squeeze()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        device = t.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=self.half, dtype=torch.float32, device=device)
            / self.half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def map_timestep(t: torch.Tensor, noise_levels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """将时间步t映射到noise_levels指定的范围。
    
    Args:
        t: 时间步索引 [B] 或 [B, 1]
        noise_levels: 时间编码映射 [n_steps]
    
    Returns:
        映射后的时间步
    """
    if noise_levels is None:
        return t.float()
    if torch.is_floating_point(t):
        return t
    t = t.squeeze()
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return noise_levels.to(t.device)[t.long()]


class ResidualBlock(nn.Module):
    """基础残差块，用于 ConvNet。"""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.activation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(inputs)
        x = self.activation2(x)
        return x


class ConvNet(nn.Module):
    """基于卷积的扩散模型，支持时间步嵌入。"""

    def __init__(self,
                 n_steps: int,
                 in_channels: int = 1,
                 intermediate_channels=None,
                 pe_dim: int = 10,
                 insert_t_to_all_layers: bool = False) -> None:
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = [10, 20, 40]
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.pe_linears = nn.ModuleList()
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, in_channels))

        self.residual_blocks = nn.ModuleList()
        prev_channel = in_channels
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = t.shape[0]
        t = self.pe(t)
        for block, linear in zip(self.residual_blocks, self.pe_linears):
            if linear is not None:
                pe = linear(t).reshape(batch_size, -1, 1, 1)
                x = x + pe
            x = block(x)
        x = self.output_layer(x)
        return x


class SeqT(nn.Module):
    """顺序容器，支持时间步嵌入传递。"""

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for module in self.mods:
            x = module(x, t_emb)
        return x


class UnetBlockDiffusion(nn.Module):
    """支持时间步嵌入的 UNet 块（用于扩散模型）。"""

    def __init__(self,
                 shape: tuple[int, int, int],
                 in_c: int,
                 out_c: int,
                 t_dim: Optional[int] = None,
                 residual: bool = False) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)
        else:
            self.residual_conv = None

        self.t_dim = t_dim
        if t_dim is not None:
            self.t_fc1 = nn.Linear(t_dim, out_c)
            self.t_fc2 = nn.Linear(t_dim, out_c)
        else:
            self.t_fc1 = None
            self.t_fc2 = None

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        if t_emb is not None:
            t_emb = t_emb.squeeze(1)
        if self.t_fc1 is not None and t_emb is not None:
            out = out + self.t_fc1(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(out)
        if self.t_fc2 is not None and t_emb is not None:
            out = out + self.t_fc2(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        if self.residual:
            out = out + self.residual_conv(x)
        out = self.activation(out)
        return out


class UNetDiffusion(nn.Module):
    """支持时间步嵌入和条件输入的 UNet（用于扩散模型）。"""

    def __init__(self,
                 n_steps: int,
                 in_channels: int = 1,
                 image_size: int = 101,
                 lr_channels: Optional[int] = None,
                 channels=None,
                 pe_dim: int = 128,
                 residual: bool = False,
                 noise_levels: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if channels is None:
            channels = [10, 20, 40, 80]
        c, h, w = in_channels, image_size, image_size
        
        # LR 通道数默认与 HR 相同,但在 TFM 模式下可能不同
        if lr_channels is None:
            lr_channels = in_channels
        
        self.in_channels = in_channels
        self.lr_channels = lr_channels
        
        # 注册noise_levels
        if noise_levels is not None:
            noise_levels = noise_levels.float()
        self.register_buffer("noise_levels", noise_levels, persistent=False)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(pe_dim, pe_dim * 4), nn.SiLU(),
            nn.Linear(pe_dim * 4, pe_dim * 4))
        self.t_dim = pe_dim * 4

        num_layers = len(channels)
        hs = [h]
        ws = [w]
        current_h, current_w = h, w
        for _ in range(num_layers - 1):
            current_h //= 2
            current_w //= 2
            hs.append(current_h)
            ws.append(current_w)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 输入是 HR + LR concatenated
        prev_channel = c + lr_channels
        for channel, ch, cw in zip(channels[:-1], hs[:-1], ws[:-1]):
            self.encoders.append(
                SeqT(
                    UnetBlockDiffusion((prev_channel, ch, cw),
                              prev_channel,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                    UnetBlockDiffusion((channel, ch, cw),
                              channel,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                ))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        mid_channel = channels[-1]
        self.mid = SeqT(
            UnetBlockDiffusion((prev_channel, hs[-1], ws[-1]),
                      prev_channel,
                      mid_channel,
                      t_dim=self.t_dim,
                      residual=residual),
            UnetBlockDiffusion((mid_channel, hs[-1], ws[-1]),
                      mid_channel,
                      mid_channel,
                      t_dim=self.t_dim,
                      residual=residual),
        )
        prev_channel = mid_channel

        for channel, ch, cw in zip(channels[-2::-1], hs[-2::-1], ws[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                SeqT(
                    UnetBlockDiffusion((channel * 2, ch, cw),
                              channel * 2,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                    UnetBlockDiffusion((channel, ch, cw),
                              channel,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                ))
            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, c, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                lr_image: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 噪声 HR 图像 [B, C, H, W]
            t: 时间步 [B] 或 [B, 1]
            lr_image: LR 条件图像 [B, lr_C, H, W]
        """
        t = map_timestep(t, self.noise_levels)
        x = torch.cat((x, lr_image), dim=1)
        t_emb = self.pe(t.squeeze())
        t_emb = self.t_mlp(t_emb)
        encoder_outs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, t_emb)
            encoder_outs.append(x)
            x = down(x)
        x = self.mid(x, t_emb)
        for decoder, up, encoder_out in zip(self.decoders, self.ups,
                                            encoder_outs[::-1]):
            x = up(x)
            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x,
                      (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                       pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x, t_emb)
        return self.conv_out(x)


# ============== 纯监督学习相关类（原 network.py）==============

class Seq(nn.Module):
    """Utility that forwards through a list of submodules."""

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules_list:
            x = module(x)
        return x


class UnetBlock(nn.Module):
    """Basic UNet block with layer norm and optional residual branch."""

    def __init__(self,
                 shape: Sequence[int],
                 in_channels: int,
                 out_channels: int,
                 residual: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.residual = residual
        if residual:
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.residual and self.shortcut is not None:
            residual = self.shortcut(residual)
        x = x + (residual if self.residual else 0)
        return self.activation(x)


class UNet(nn.Module):
    """UNet that maps LR inputs to HR predictions directly."""

    def __init__(self,
                 in_channels: int,
                 lr_channels: int,
                 image_size: int,
                 channels: Sequence[int],
                 residual_blocks: bool = False) -> None:
        super().__init__()
        c, h, w = in_channels, image_size, image_size
        self.in_channels = in_channels
        self.lr_channels = lr_channels

        num_layers = len(channels)
        heights = [h]
        widths = [w]
        cur_h, cur_w = h, w
        for _ in range(num_layers - 1):
            cur_h //= 2
            cur_w //= 2
            heights.append(cur_h)
            widths.append(cur_w)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev_channels = lr_channels
        for channel, ch, cw in zip(channels[:-1], heights[:-1], widths[:-1]):
            self.encoders.append(
                Seq(
                    UnetBlock((prev_channels, ch, cw),
                              prev_channels,
                              channel,
                              residual=residual_blocks),
                    UnetBlock((channel, ch, cw),
                              channel,
                              channel,
                              residual=residual_blocks),
                ))
            self.downs.append(nn.Conv2d(channel, channel, kernel_size=2, stride=2))
            prev_channels = channel

        mid_channels = channels[-1]
        self.mid = Seq(
            UnetBlock((prev_channels, heights[-1], widths[-1]),
                      prev_channels,
                      mid_channels,
                      residual=residual_blocks),
            UnetBlock((mid_channels, heights[-1], widths[-1]),
                      mid_channels,
                      mid_channels,
                      residual=residual_blocks),
        )
        prev_channels = mid_channels

        for channel, ch, cw in zip(channels[-2::-1], heights[-2::-1], widths[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(prev_channels, channel, kernel_size=2, stride=2))
            self.decoders.append(
                Seq(
                    UnetBlock((channel * 2, ch, cw),
                              channel * 2,
                              channel,
                              residual=residual_blocks),
                    UnetBlock((channel, ch, cw),
                              channel,
                              channel,
                              residual=residual_blocks),
                ))
            prev_channels = channel

        self.head = nn.Conv2d(prev_channels, c, kernel_size=3, padding=1)

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        x = lr_image
        encoder_outs: List[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encoder_outs.append(x)
            x = down(x)
        x = self.mid(x)
        for decoder, up, encoder_out in zip(self.decoders, self.ups, encoder_outs[::-1]):
            x = up(x)
            pad_h = encoder_out.shape[2] - x.shape[2]
            pad_w = encoder_out.shape[3] - x.shape[3]
            if pad_h or pad_w:
                x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                              pad_h // 2, pad_h - pad_h // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x)
        return self.head(x)


# ============== 模型配置字典 ==============

convnet_small_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 20],
    'pe_dim': 128,
}

convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True,
}

convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True,
}

unet_1_cfg = {
    'type': 'UNetDiffusion',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128
}

unet_res_cfg = {
    "type": "UNet",
    "channels": [16, 32, 64, 128, 256],
    "residual": True,
}

unet_res_diffusion_cfg = {
    'type': 'UNetDiffusion',
    'channels': [16, 32, 64, 128, 256],
    'pe_dim': 128,
    'residual': True,
}


# ============== 模型构建函数 ==============

def build_network(config: dict,
                  in_channels: int,
                  image_size: int,
                  lr_channels: int,
                  n_steps: Optional[int] = None,
                  noise_levels: Optional[torch.Tensor] = None) -> nn.Module:
    """
    构建网络模型。
    
    Args:
        config: 模型配置字典
        in_channels: 输出通道数（HR 图像通道数）
        image_size: 图像尺寸
        lr_channels: 输入通道数（LR 条件图像通道数）
        n_steps: 扩散步数（仅扩散模型需要）
        noise_levels: 时间编码映射（用于某些扩散模型，如I2SB）
    
    Returns:
        构建好的网络模型
    """
    cfg = config.copy()
    network_type = cfg.pop('type', 'UNet')
    
    # 添加公共参数
    cfg['in_channels'] = in_channels
    cfg['image_size'] = image_size
    
    if network_type == 'ConvNet':
        if n_steps is None:
            raise ValueError("ConvNet requires n_steps parameter")
        return ConvNet(n_steps, **cfg)
    
    elif network_type == 'UNetDiffusion':
        # 扩散模型的 UNet
        if n_steps is None:
            raise ValueError("UNetDiffusion requires n_steps parameter")
        cfg['lr_channels'] = lr_channels
        cfg['noise_levels'] = noise_levels
        return UNetDiffusion(n_steps, **cfg)
    
    elif network_type == 'UNet':
        # 纯监督学习的 UNet
        channels = cfg.get("channels", [16, 32, 64, 128, 256])
        residual = bool(cfg.get("residual", False))
        return UNet(in_channels=in_channels,
                    lr_channels=lr_channels,
                    image_size=image_size,
                    channels=channels,
                    residual_blocks=residual)
    
    else:
        raise KeyError(f"Unsupported network type '{network_type}'. "
                       f"Available: ConvNet, UNetDiffusion, UNet")
