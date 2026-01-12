# Repository Guidelines

## Project Structure & Module Organization
The codebase is now a plain supervised SR pipeline: `unetonly_train.py` drives UNet training, while the backbone lives in `network.py`. Supporting pieces include `dataset.py`, `attention.py`, `logger.py`, plus shared utilities. Evaluation uses `eval.py`. Default configuration JSON files (`train.json`, `eval.json`) sit in the repository root, and artifacts are still written under `SR/`, `work_dirs/`, or a user-provided folder.

## Build, Test, and Development Commands
Launch training with `python unetonly_train.py --config train.json`. Update dataset paths and logging knobs directly within that JSON—避免在脚本里硬编码。Evaluate checkpoints with `python eval.py --config eval.json` to export PSNR/SSIM CSVs and per-image predictions. TensorBoard logging is unchanged (`tensorboard --logdir runs`).

## Coding Style & Naming Conventions
Follow PEP 8 (four-space indentation, snake_case for functions/variables). Keep config loaders and CLI entry points lightweight, leverage helpers from modules such as `dataset.py` and `network.py`, and prefer pure functions for shared logic. Provide concise type hints for public APIs (for example, `cfg: Dict[str, Any]`, `device: torch.device`) and add short comments when tensor shapes or units are not obvious. Maintain import order compatible with `isort`/`black`; no trailing whitespace and stick to ASCII unless the surrounding file already uses UTF-8 text.

## Testing Guidelines
A lightweight pytest suite now lives in `tests/`. Run `pytest` before merging以确保数据管线和张量形状假设保持正确。For end-to-end validation, execute `python eval.py --config eval.json` and capture PSNR/SSIM from the generated CSV. When adding new utilities, include shape assertions or unit tests to avoid silent regressions.

## Commit & Pull Request Guidelines
Use imperative present-tense commit subjects around 60 characters (for example, `feat: add JSON configs for SR training`). Reference related experiments or issues in the body, and list commands executed (training, evaluation, pytest). Pull requests should summarise motivation, highlight key files touched, outline testing status, and attach relevant outputs (metrics table or preview image path) before requesting review.

## Data & Configuration Notes
Training assumes CUDA by default; set `"device": "cpu"` and `"num_workers": 0` in `train.json` for CPU-only runs. Always duplicate model config templates with `.copy()` before modifying them to avoid shared-state bugs. Store checkpoints under `SR/` (tracked via JSON configs) and keep large binaries ignored. All imports remain local—ensure scripts execute from the repository root or prepend `Path(__file__).parent` to `PYTHONPATH` when launching notebooks.
