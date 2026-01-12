# UNetonly-sr-for-TFM

本项目面向超声波全聚焦成像超分辨（TFM SR）任务，实现了一个简单直接的单 UNet 训练/评估流水线（LR→HR 的监督学习），涵盖数据加载、模型构建、训练脚本与指标导出工具。

---

## 文档索引

- 项目架构：[docs/architecture.md](docs/architecture.md)
- 数据准备：[docs/data_preparation.md](docs/data_preparation.md)
- 训练指南：[docs/training_guide.md](docs/training_guide.md)
- 评估流程：[docs/evaluation_guide.md](docs/evaluation_guide.md)
- 配置说明：[docs/configuration.md](docs/configuration.md)
- API 速查：[docs/api_reference.md](docs/api_reference.md)
- 实验记录：[docs/experiments.md](docs/experiments.md)
- 常见问题：[docs/faq.md](docs/faq.md)
- 教程 Notebook：[docs/tutorials](docs/tutorials)

---

## 数据集采集

- 默认数据由 k-Wave 仿真生成，可参考开源脚本：<https://github.com/cacdcaecawae/data-for-TFM-kwave>。
- 仿真后将低分辨率观测（`TFM/样本号`）与高分辨率真值（`hr/样本号`）整理为 HDF5，即可被 `dataset.H5PairedDataset` 自动加载。
- HDF5 目录结构、归一化及增强策略详见 [数据准备说明](docs/data_preparation.md)，也可结合 [项目架构](docs/architecture.md) 了解数据如何贯穿网络。

## 训练流程

1. **环境准备**  
   Python 3.10+，PyTorch ≥ 2.1（建议 GPU 版本），常用依赖包括 `torchmetrics`、`opencv-python`、`tensorboard`、`einops`、`tqdm` 等。  
   ```bash
   pip install torch torchvision torchmetrics opencv-python tensorboard einops tqdm
   ```
2. **配置编辑**  
   修改根目录下的 `train.json`，设置数据路径、UNet 配置和日志参数。完整字段说明见 [配置文件详解](docs/configuration.md)。
3. **启动训练**  
   ```bash
   python unetonly_train.py --config train.json
   ```
   - 训练脚本会构建 `H5PairedDataset` 并以 MSE 损失直接拟合 HR，配合余弦调度与 EMA 更新权重。
   - 日志写入 `runs/`，可通过 `tensorboard --logdir runs` 查看；若开启 `logging.save_preview_images`，会在 `SR/previews/` 导出中间预测。
4. 更多调参建议（AMP、EMA、数据增强等）可参考 [训练指南](docs/training_guide.md) 与教程 `docs/tutorials/quickstart.ipynb`。

## 评估流程

1. **准备配置**  
   定制根目录下的 `eval.json`，填入评估集路径、模型 checkpoint、阈值等超参。
2. **执行评估**  
   ```bash
   python eval.py --config eval.json
   ```
   - `eval.py` 会按配置加载模型，直接前向生成超分结果，并计算 PSNR / SSIM。
   - 输出目录（`output.root`）默认包含预测图像、`results.txt` 与 `results_per_image.csv`。
3. 阈值裁剪说明详见 [评估指南](docs/evaluation_guide.md) 与 [API 速查](docs/api_reference.md)。

## 其他说明

- 如未搭建 Python 环境，建议使用 “VS Code + Anaconda” 组合，可参考入门视频：[py 虚拟环境搭建](https://www.bilibili.com/video/BV1bQ4y1n7sn/?spm_id_from=333.1387.favlist.content.click&vd_source=fa7df163675956e625633678e8c906fa)。
- 推荐配合 DeepWiki 学习 SR/反演相关知识：  
  - DeepWiki 主页：<https://deepwiki.org/>（可直接粘贴仓库地址，例如 `https://github.com/cacdcaecawae/ddpm-for-srtfm/tree/hdf5`）。  
  - DeepWiki 使用介绍：<https://zhuanlan.zhihu.com/p/1900126204851381576>。
- 提交前建议执行 `pytest`（位于 `tests/`），并将实验配置、指标记录在 [实验记录与结果](docs/experiments.md)；常见问题可参考 [常见问题](docs/faq.md)。
- 教程 Notebook：  
  - [docs/tutorials/quickstart.ipynb](docs/tutorials/quickstart.ipynb)：验证环境、读取配置并生成训练/评估命令。
