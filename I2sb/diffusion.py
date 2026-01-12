# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var
    
class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev
    
    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0
    
    def pred_x0_fn(self, xt, step, net, cond, clip_denoise=False):
        """预测 x0，支持条件输入 (cond为lr image)"""
        step_tensor = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)
        out = net(xt, step_tensor, cond)
        return self.compute_pred_x0(step_tensor, xt, out, clip_denoise=clip_denoise)
    
    def ddpm_sampling(self, steps, net, x1, ot_ode=False, log_steps=None, verbose=True, return_trajectory=False):
        """
        DDPM 采样方法
        
        Args:
            steps: 升序时间步序列，例如 [0, 10, 20, ..., T]
            net: 去噪网络
            x1: 起始噪声 (对应 x_T)
            lr_image: 条件图像 (LR 图像)
            ot_ode: 是否使用 OT-ODE（确定性采样）
            log_steps: 记录哪些时间步的中间结果
            verbose: 是否显示进度条
            return_trajectory: 是否返回完整轨迹
        """
        assert isinstance(steps, (list, np.ndarray)), "steps 必须是列表或数组"
        assert len(steps) >= 2, "steps 至少需要包含2个时间步"
        assert steps[0] == 0, "steps 必须从 0 开始"
        
        xt = x1[:,0:1].detach().to(self.device)
        
        if return_trajectory:
            xs = []
            pred_x0s = []
            log_steps = log_steps or steps
        
        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        
        for prev_step, step in pair_steps:
            pred_x0 = self.pred_x0_fn(xt, step, net, cond=x1)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            
            if return_trajectory and prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
        
        if return_trajectory:
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return xt, stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
        else:
            return xt  # 只返回最终结果
