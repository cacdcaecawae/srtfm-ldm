import torch

from network import network as base_net
from network import networksb as sb_net


def test_unet_forward_shape() -> None:
    cfg = {"type": "UNet", "channels": [4, 8], "residual": False}
    net = base_net.build_network(cfg, in_channels=1, image_size=16, lr_channels=3)
    lr_image = torch.randn(2, 3, 16, 16)
    output = net(lr_image)
    assert output.shape == (2, 1, 16, 16)


def test_unet_diffusion_forward_shape() -> None:
    cfg = {"type": "UNetDiffusion", "channels": [4, 8], "pe_dim": 16, "residual": False}
    net = base_net.build_network(cfg,
                                 in_channels=1,
                                 image_size=16,
                                 lr_channels=4,
                                 n_steps=10)
    x_t = torch.randn(2, 1, 16, 16)
    condition = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 10, (2,))
    output = net(x_t, t, condition)
    assert output.shape == (2, 1, 16, 16)


def test_networksb_unet_diffusion_shape() -> None:
    cfg = {"type": "UNetDiffusion", "channels": [4, 8], "pe_dim": 16, "residual": False}
    noise_levels = torch.linspace(0.1, 1.0, 10)
    net = sb_net.build_network(cfg,
                               in_channels=1,
                               image_size=16,
                               lr_channels=4,
                               n_steps=10,
                               noise_levels=noise_levels)
    x_t = torch.randn(2, 1, 16, 16)
    condition = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 10, (2,))
    output = net(x_t, t, condition)
    assert output.shape == (2, 1, 16, 16)
