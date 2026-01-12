"""Simple VAE finetune script.

Example (PowerShell):
    python vae_finetune.py `
      --h5_path data/output_merge/train.h5 `
      --vae_path SR/sd-vae-ft-mse `
      --output_dir SR/vae_ft `
      --epochs 5 `
      --batch_size 2 `
      --lr 1e-5 `
      --kl_weight 1e-6

Example (HR + I only):
    python vae_finetune.py `
      --h5_path data/output_merge/train.h5 `
      --vae_path SR/sd-vae-ft-mse `
      --output_dir SR/vae_ft `
      --no_use_tfm_xy
"""
import argparse
import contextlib
import random
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers.models import AutoencoderKL


class H5VaeDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        transpose_hr: bool = True,
        transpose_lr: bool = True,
        use_hr: bool = True,
        use_tfm_i: bool = True,
        use_tfm_xy: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.h5_path = h5_path
        self.transpose_hr = transpose_hr
        self.transpose_lr = transpose_lr
        self.use_hr = use_hr
        self.use_tfm_i = use_tfm_i
        self.use_tfm_xy = use_tfm_xy
        self.max_samples = max_samples

        self._file: Optional[h5py.File] = None
        self._index: Optional[List[Tuple[str, str]]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None

    def _open_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    @staticmethod
    def _read_node(node: h5py.Group) -> np.ndarray:
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        if "data" in node:
            return np.asarray(node["data"])
        for key in node.keys():
            if isinstance(node[key], h5py.Dataset):
                return np.asarray(node[key])
        raise ValueError("No dataset found in node.")

    def _init_index(self) -> None:
        if self._index is not None:
            return

        f = self._open_file()
        if "hr" not in f or "TFM" not in f:
            raise KeyError("H5 file must contain 'hr' and 'TFM' groups.")

        hr_names = set(f["hr"].keys())
        tfm_names = set(f["TFM"].keys())
        shared = sorted(hr_names & tfm_names)
        if not shared:
            raise ValueError("No shared samples between hr and TFM groups.")
        if self.max_samples is not None:
            shared = shared[: self.max_samples]

        index: List[Tuple[str, str]] = []
        for name in shared:
            if self.use_hr:
                index.append((name, "hr"))
            if self.use_tfm_i:
                index.append((name, "I"))
            if self.use_tfm_xy:
                index.append((name, "X"))
                index.append((name, "Y"))

        if not index:
            raise ValueError("No data sources selected for training.")

        self._index = index

    @staticmethod
    def _maybe_transpose(array: np.ndarray, transpose: bool) -> np.ndarray:
        if not transpose:
            return array
        if array.ndim == 2:
            return array.T
        if array.ndim == 3:
            return np.transpose(array, (0, 2, 1))
        return array

    @staticmethod
    def _normalize_to_minus_one_one(array: np.ndarray) -> torch.Tensor:
        if array.ndim == 2:
            array = array[None, ...]
        tensor = torch.from_numpy(array).float()
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
        tensor = tensor * 2.0 - 1.0
        return tensor

    def _load_array(self, name: str, kind: str) -> np.ndarray:
        f = self._open_file()
        if kind == "hr":
            return self._read_node(f["hr"][name])

        tfm_group = f["TFM"][name]
        if kind == "I":
            if isinstance(tfm_group, h5py.Dataset):
                return np.asarray(tfm_group)
            if "I" in tfm_group:
                return np.asarray(tfm_group["I"])
            if "intensity" in tfm_group:
                return np.asarray(tfm_group["intensity"])
            return self._read_node(tfm_group)
        if kind == "X":
            return np.asarray(tfm_group["X"])
        if kind == "Y":
            return np.asarray(tfm_group["Y"])
        raise ValueError(f"Unknown kind: {kind}")

    def __len__(self) -> int:
        self._init_index()
        return len(self._index)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._init_index()
        name, kind = self._index[idx]  # type: ignore[index]
        array = self._load_array(name, kind)
        if kind == "hr":
            array = self._maybe_transpose(array, self.transpose_hr)
        else:
            array = self._maybe_transpose(array, self.transpose_lr)
        tensor = self._normalize_to_minus_one_one(array)
        if tensor.shape[0] != 1:
            tensor = tensor[:1]
        tensor = tensor.repeat(3, 1, 1)
        h, w = tensor.shape[-2], tensor.shape[-1]
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError("Input size must be divisible by 8. Please pad or crop.")
        return tensor

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple VAE finetune script")
    parser.add_argument("--h5_path", required=True, help="Path to h5 file")
    parser.add_argument("--vae_path", required=True, help="Path to pretrained VAE")
    parser.add_argument("--output_dir", required=True, help="Where to save finetuned VAE")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_hr", action="store_true", default=True)
    parser.add_argument("--use_tfm_i", action="store_true", default=True)
    parser.add_argument("--use_tfm_xy", action="store_true", default=True)
    parser.add_argument("--no_use_hr", dest="use_hr", action="store_false")
    parser.add_argument("--no_use_tfm_i", dest="use_tfm_i", action="store_false")
    parser.add_argument("--no_use_tfm_xy", dest="use_tfm_xy", action="store_false")
    parser.add_argument("--transpose_hr", action="store_true", default=True)
    parser.add_argument("--transpose_lr", action="store_true", default=True)
    parser.add_argument("--no_transpose_hr", dest="transpose_hr", action="store_false")
    parser.add_argument("--no_transpose_lr", dest="transpose_lr", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = H5VaeDataset(
        h5_path=args.h5_path,
        transpose_hr=args.transpose_hr,
        transpose_lr=args.transpose_lr,
        use_hr=args.use_hr,
        use_tfm_i=args.use_tfm_i,
        use_tfm_xy=args.use_tfm_xy,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.to(device).train()

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext

    print(f"samples: {len(dataset)}")
    print(f"device: {device}")

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(enabled=args.amp and device.type == "cuda"):
                posterior = vae.encode(batch)
                z = posterior.sample()
                recon = vae.decode(z).sample
                recon_loss = F.mse_loss(recon, batch)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + args.kl_weight * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()

            if args.log_interval > 0 and step % args.log_interval == 0:
                avg_loss = running_loss / step
                print(
                    f"epoch {epoch} step {step}/{len(loader)} "
                    f"loss={avg_loss:.6g} recon={running_recon/step:.6g} kl={running_kl/step:.6g}"
                )

        avg_loss = running_loss / max(1, len(loader))
        avg_recon = running_recon / max(1, len(loader))
        avg_kl = running_kl / max(1, len(loader))
        print(
            f"epoch {epoch} done loss={avg_loss:.6g} recon={avg_recon:.6g} kl={avg_kl:.6g}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(output_dir)
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    train()

