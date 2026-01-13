import h5py
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from pathlib import Path
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); vae_path = "./SR/vae_ft"; h5_path = "./data/output_merge/train.h5"; out_dir = Path("./vae_compression_vis")
transpose_hr = True; transpose_lr = True
scale_probe_samples = 1500

def norm_single(x):
    mn, mx = x.min(), x.max(); x = (x - mn) / (mx - mn + 1e-8)
    return x * 2 - 1, mn, mx

def denorm(x, mn, mx):
    return (x + 1) / 2 * (mx - mn) + mn

def roundtrip(vae, x):
    with torch.no_grad():
        posterior = vae.encode(x.to(device)).latent_dist
        z = posterior.mode()
        y = vae.decode(z).sample
        kl = posterior.kl().mean().item()
    return y.cpu(), z.cpu(), kl

def save_pair(path, orig, recon):
    orig = orig.detach().cpu().numpy(); recon = recon.detach().cpu().numpy(); mn, mx = orig.min(), orig.max()
    if mx - mn < 1e-8: o = np.zeros_like(orig); r = np.zeros_like(recon)
    else: o = (orig - mn) / (mx - mn); r = (recon - mn) / (mx - mn)
    img = np.concatenate([(o * 255).clip(0, 255), (r * 255).clip(0, 255)], 1).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)

def process_single(vae, arr2d, tag, out_dir, scale):
    x = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    x_n, x_min, x_max = norm_single(x)
    x_rec, z, kl = roundtrip(vae, x_n)
    z_scaled = z * scale
    z_mean = z_scaled.mean().item()
    z_std = z_scaled.std().item()
    kl_per_elem = kl / z.numel()
    x_out = denorm(x_rec, x_min, x_max)
    mse_raw = (x_out - x).pow(2).mean().item()
    mse_norm = (x_rec - x_n).pow(2).mean().item()
    save_pair(out_dir / f"{tag}.png", x[0, 0], x_out[0, 0])
    return mse_raw, mse_norm, z, x_min.item(), x_max.item(), kl, kl_per_elem, z_mean, z_std

def make_tensor_1ch(arr2d):
    x = torch.from_numpy(arr2d).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    x = x * 2 - 1
    x = x[:1].repeat(3, 1, 1)
    return x

with h5py.File(h5_path, "r") as f:
    name = sorted(f["hr"].keys())[120]; hr_node = f["hr"][name]
    hr = hr_node[:] if isinstance(hr_node, h5py.Dataset) else (hr_node["data"][:] if "data" in hr_node else hr_node[list(hr_node.keys())[0]][:])
    if transpose_hr: hr = hr.T
    tfm = f["TFM"][name]
    intensity = tfm["I"][:] if "I" in tfm else tfm["intensity"][:]
    x_coord = tfm["X"][:]; y_coord = tfm["Y"][:]
    if transpose_lr: intensity, x_coord, y_coord = intensity.T, x_coord.T, y_coord.T

vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval(); out_dir.mkdir(parents=True, exist_ok=True)
scale_cfg = float(getattr(vae.config, "scaling_factor", 1.0))

hr_mse_raw, hr_mse_norm, hr_z, hr_min, hr_max, hr_kl, hr_kl_elem, hr_z_mean, hr_z_std = process_single(vae, hr, f"{name}_hr", out_dir, scale_cfg)
i_mse_raw, i_mse_norm, i_z, i_min, i_max, i_kl, i_kl_elem, i_z_mean, i_z_std = process_single(vae, intensity, f"{name}_tfm_I", out_dir, scale_cfg)
x_mse_raw, x_mse_norm, x_z, x_min, x_max, x_kl, x_kl_elem, x_z_mean, x_z_std = process_single(vae, x_coord, f"{name}_tfm_X", out_dir, scale_cfg)
y_mse_raw, y_mse_norm, y_z, y_min, y_max, y_kl, y_kl_elem, y_z_mean, y_z_std = process_single(vae, y_coord, f"{name}_tfm_Y", out_dir, scale_cfg)

print("sample:", name)
print("HR in", hr.shape, "latent", tuple(hr_z.shape), "min/max", f"{hr_min:.6g}", f"{hr_max:.6g}", "mse_raw", f"{hr_mse_raw:.6g}", "mse_norm", f"{hr_mse_norm:.6g}", "kl", f"{hr_kl:.6g}", "kl_elem", f"{hr_kl_elem:.6g}", "z_mean", f"{hr_z_mean:.6g}", "z_std", f"{hr_z_std:.6g}")
print("TFM I", intensity.shape, "latent", tuple(i_z.shape), "min/max", f"{i_min:.6g}", f"{i_max:.6g}", "mse_raw", f"{i_mse_raw:.6g}", "mse_norm", f"{i_mse_norm:.6g}", "kl", f"{i_kl:.6g}", "kl_elem", f"{i_kl_elem:.6g}", "z_mean", f"{i_z_mean:.6g}", "z_std", f"{i_z_std:.6g}")
print("TFM X", x_coord.shape, "latent", tuple(x_z.shape), "min/max", f"{x_min:.6g}", f"{x_max:.6g}", "mse_raw", f"{x_mse_raw:.6g}", "mse_norm", f"{x_mse_norm:.6g}", "kl", f"{x_kl:.6g}", "kl_elem", f"{x_kl_elem:.6g}", "z_mean", f"{x_z_mean:.6g}", "z_std", f"{x_z_std:.6g}")
print("TFM Y", y_coord.shape, "latent", tuple(y_z.shape), "min/max", f"{y_min:.6g}", f"{y_max:.6g}", "mse_raw", f"{y_mse_raw:.6g}", "mse_norm", f"{y_mse_norm:.6g}", "kl", f"{y_kl:.6g}", "kl_elem", f"{y_kl_elem:.6g}", "z_mean", f"{y_z_mean:.6g}", "z_std", f"{y_z_std:.6g}")

with torch.no_grad():
    with h5py.File(h5_path, "r") as f:
        names = sorted(f["hr"].keys())
        count = min(scale_probe_samples, len(names))
        zs = []
        for idx in range(count):
            name_i = names[idx]
            hr_node = f["hr"][name_i]
            hr_arr = hr_node[:] if isinstance(hr_node, h5py.Dataset) else (hr_node["data"][:] if "data" in hr_node else hr_node[list(hr_node.keys())[0]][:])
            if transpose_hr: hr_arr = hr_arr.T
            x = make_tensor_1ch(hr_arr).unsqueeze(0).to(device)
            z = vae.encode(x).latent_dist.sample()
            zs.append(z.cpu())
        z_all = torch.cat(zs, dim=0)
        z_std = z_all.std().item()
        scale_reco = 1.0 / z_std if z_std > 0 else 1.0

print("scale_probe_samples:", scale_probe_samples)
print("z_std(sample):", f"{z_std:.6g}", "recommended_scale:", f"{scale_reco:.6g}")
print("saved:", str(out_dir))
