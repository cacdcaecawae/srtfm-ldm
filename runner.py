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
    ("train", "ddpm-single"): ROOT / "ddpm" / "ddpm_train.py",
    ("eval", "ddpm-single"): ROOT / "ddpm" / "ddpm_eval.py",
    ("train", "i2sb"): ROOT / "I2sb" / "i2sb_train.py",
    ("eval", "i2sb"): ROOT / "I2sb" / "i2sb_eval.py",
    ("train", "i2sb-single"): ROOT / "I2sb" / "i2sb_train.py",
    ("eval", "i2sb-single"): ROOT / "I2sb" / "i2sb_eval.py",
}

SINGLE_STAGE_PIPELINES = {"ddpm-single", "i2sb-single"}

EXAMPLES = """Examples:
  python runner.py train unet
  python runner.py train ddpm
  python runner.py train ddpm-single
  python runner.py train i2sb
  python runner.py train i2sb-single
  Use 'eval' in place of 'train' to run evaluation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a training or evaluation pipeline with default configs.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        choices=("unet", "ddpm", "ddpm-single", "i2sb", "i2sb-single"),
        help="Choose which pipeline to run.",
    )
    return parser.parse_args()


def run_script(script_path: Path, single_stage: bool) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    old_argv = sys.argv
    old_cwd = Path.cwd()
    old_single_stage = os.environ.get("SR_SINGLE_STAGE")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        if single_stage:
            os.environ["SR_SINGLE_STAGE"] = "1"
        else:
            os.environ.pop("SR_SINGLE_STAGE", None)
        os.chdir(ROOT)
        sys.argv = [str(script_path)]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        if old_single_stage is None:
            os.environ.pop("SR_SINGLE_STAGE", None)
        else:
            os.environ["SR_SINGLE_STAGE"] = old_single_stage


def main() -> None:
    args = parse_args()
    single_stage = args.pipeline in SINGLE_STAGE_PIPELINES
    script_path = PIPELINE_SCRIPTS[(args.mode, args.pipeline)]
    run_script(script_path, single_stage=single_stage)


if __name__ == "__main__":
    main()

# Examples:
# python runner.py train unet
# python runner.py train ddpm
# python runner.py train ddpm-single
# python runner.py train i2sb
# python runner.py train i2sb-single
# Use 'eval' in place of 'train' to run evaluation.
