from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent

PIPELINE_SCRIPTS: Dict[Tuple[str, str], Path] = {
    ("train", "unet"): ROOT / "unetbase" / "unetonly_train.py",
    ("eval", "unet"): ROOT / "eval.py",
    ("train", "ddpm"): ROOT / "ddpm" / "ddpm_train.py",
    ("eval", "ddpm"): ROOT / "ddpm" / "ddpm_eval.py",
    ("train", "i2sb"): ROOT / "I2sb" / "i2sb_train.py",
    ("eval", "i2sb"): ROOT / "I2sb" / "i2sb_eval.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a training or evaluation pipeline with default configs."
    )
    parser.add_argument(
        "mode",
        type=str.lower,
        choices=("train", "eval"),
        help="Choose whether to train or evaluate.",
    )
    parser.add_argument(
        "pipeline",
        type=str.lower,
        choices=("unet", "ddpm", "i2sb"),
        help="Choose which pipeline to run.",
    )
    return parser.parse_args()


def run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    old_argv = sys.argv
    old_cwd = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        os.chdir(ROOT)
        sys.argv = [str(script_path)]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def main() -> None:
    args = parse_args()
    script_path = PIPELINE_SCRIPTS[(args.mode, args.pipeline)]
    run_script(script_path)


if __name__ == "__main__":
    main()
