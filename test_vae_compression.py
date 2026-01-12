import h5py
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from pathlib import Path
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); vae_path = "./SR/sd-vae-ft-mse"; h5_path = "./data/output_merge/eval.h5"; out_dir = Path("./vae_compression_vis")
transpose_hr = True; transpose_lr = True

def norm_single(x):
    mn, mx = x.min(), x.max(); x = (x - mn) / (mx - mn + 1e-8)
    return x * 2 - 1, mn, mx

def denorm(x, mn, mx):
    return (x + 1) / 2 * (mx - mn) + mn

def roundtrip(vae, x):
    with torch.no_grad(): z = vae.encode(x.to(device)).latent_dist.mode(); y = vae.decode(z).sample
    return y.cpu(), z.cpu()

def save_pair(path, orig, recon):
    orig = orig.detach().cpu().numpy(); recon = recon.detach().cpu().numpy(); mn, mx = orig.min(), orig.max()
    if mx - mn < 1e-8: o = np.zeros_like(orig); r = np.zeros_like(recon)
    else: o = (orig - mn) / (mx - mn); r = (recon - mn) / (mx - mn)
    img = np.concatenate([(o * 255).clip(0, 255), (r * 255).clip(0, 255)], 1).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)

def process_single(vae, arr2d, tag, out_dir):
    x = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    x_n, x_min, x_max = norm_single(x)
    x_rec, z = roundtrip(vae, x_n)
    x_out = denorm(x_rec, x_min, x_max)
    mse_raw = (x_out - x).pow(2).mean().item()
    mse_norm = (x_rec - x_n).pow(2).mean().item()
    save_pair(out_dir / f"{tag}.png", x[0, 0], x_out[0, 0])
    return mse_raw, mse_norm, z, x_min.item(), x_max.item()

with h5py.File(h5_path, "r") as f:
    name = sorted(f["hr"].keys())[80]; hr_node = f["hr"][name]
    hr = hr_node[:] if isinstance(hr_node, h5py.Dataset) else (hr_node["data"][:] if "data" in hr_node else hr_node[list(hr_node.keys())[0]][:])
    if transpose_hr: hr = hr.T
    tfm = f["TFM"][name]
    intensity = tfm["I"][:] if "I" in tfm else tfm["intensity"][:]
    x_coord = tfm["X"][:]; y_coord = tfm["Y"][:]
    if transpose_lr: intensity, x_coord, y_coord = intensity.T, x_coord.T, y_coord.T

vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval(); out_dir.mkdir(parents=True, exist_ok=True)

hr_mse_raw, hr_mse_norm, hr_z, hr_min, hr_max = process_single(vae, hr, f"{name}_hr", out_dir)
i_mse_raw, i_mse_norm, i_z, i_min, i_max = process_single(vae, intensity, f"{name}_tfm_I", out_dir)
x_mse_raw, x_mse_norm, x_z, x_min, x_max = process_single(vae, x_coord, f"{name}_tfm_X", out_dir)
y_mse_raw, y_mse_norm, y_z, y_min, y_max = process_single(vae, y_coord, f"{name}_tfm_Y", out_dir)

print("sample:", name)
print("HR in", hr.shape, "latent", tuple(hr_z.shape), "min/max", f"{hr_min:.6g}", f"{hr_max:.6g}", "mse_raw", f"{hr_mse_raw:.6g}", "mse_norm", f"{hr_mse_norm:.6g}")
print("TFM I", intensity.shape, "latent", tuple(i_z.shape), "min/max", f"{i_min:.6g}", f"{i_max:.6g}", "mse_raw", f"{i_mse_raw:.6g}", "mse_norm", f"{i_mse_norm:.6g}")
print("TFM X", x_coord.shape, "latent", tuple(x_z.shape), "min/max", f"{x_min:.6g}", f"{x_max:.6g}", "mse_raw", f"{x_mse_raw:.6g}", "mse_norm", f"{x_mse_norm:.6g}")
print("TFM Y", y_coord.shape, "latent", tuple(y_z.shape), "min/max", f"{y_min:.6g}", f"{y_max:.6g}", "mse_raw", f"{y_mse_raw:.6g}", "mse_norm", f"{y_mse_norm:.6g}")
print("saved:", str(out_dir))
