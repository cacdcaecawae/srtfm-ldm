# UNet SR for TFM

本项目面向超声波全聚焦成像超分辨（TFM SR）任务，提供完整的深度学习超分辨流水线，包含多种模型架构（UNet、DDPM、I2SB）和训练策略（单阶段/两阶段）。

---

## 项目结构

```
unet-srtfm/
├── runner.py                 # 统一训练/评估入口
├── dataset.py                # 数据加载与预处理
├── logger.py                 # 日志记录工具
├── conftest.py               # pytest 配置
├── requirements.txt          # Python 依赖
├── AGENTS.md                 # 项目规范与开发指南
│
├── unetbase/                 # UNet 监督学习管线
│   ├── unetonly_train.py     # UNet 训练脚本
│   ├── train.json            # 训练配置
│   └── eval.json             # 评估配置
│
├── ddpm/                     # DDPM 扩散模型管线
│   ├── ddpm_train.py         # DDPM 训练脚本
│   ├── ddpm_eval.py          # DDPM 评估脚本
│   ├── ddpm.py               # DDPM 算法实现
│   ├── ddpm_baseline.json    # 训练配置
│   └── ddpm_eval.json        # 评估配置
│
├── I2sb/                     # I2SB 薛定谔桥管线
│   ├── i2sb_train.py         # I2SB 训练脚本
│   ├── i2sb_eval.py          # I2SB 评估脚本
│   ├── diffusion.py          # I2SB 算法实现
│   ├── i2sb_twostage.json    # 训练配置
│   └── eval.json             # 评估配置
│
├── network/                  # 网络架构定义
│   ├── network.py            # 基础 UNet 实现
│   └── networksb.py          # 扩散模型 UNet
│
├── tests/                    # 单元测试
│   ├── test_dataset.py       # 数据集测试
│   └── test_networks.py      # 网络测试
│
└── runs/                     # 训练日志和 TensorBoard 输出
```

---

## 快速开始

### 环境准备
```bash
pip install -r requirements.txt
```

要求：Python 3.10+，PyTorch ≥ 2.1（建议 GPU 版本）

### 可用管线

本项目提供 **5 种训练管线**，通过 `runner.py` 统一调用：

| 管线 | 说明 | 训练命令 | 评估命令 |
|------|------|---------|---------|
| **unet** | 监督学习 UNet（LR→HR） | `python runner.py train unet` | `python runner.py eval unet` |
| **ddpm** | 两阶段 DDPM（UNet→DDPM 细化） | `python runner.py train ddpm` | `python runner.py eval ddpm` |
| **ddpm-single** | 单阶段 DDPM（直接从 LR 生成） | `python runner.py train ddpm-single` | `python runner.py eval ddpm-single` |
| **i2sb** | 两阶段 I2SB（UNet→桥接细化） | `python runner.py train i2sb` | `python runner.py eval i2sb` |
| **i2sb-single** | 单阶段 I2SB（直接桥接） | `python runner.py train i2sb-single` | `python runner.py eval i2sb-single` |

### 配置文件

每个管线都有对应的配置文件（JSON 格式）：
- UNet: `unetbase/train.json`, `unetbase/eval.json`
- DDPM: `ddpm/ddpm_baseline.json`, `ddpm/ddpm_eval.json`
- I2SB: `I2sb/i2sb_twostage.json`, `I2sb/eval.json`

修改配置文件中的数据路径、模型参数等即可自定义训练。

### 数据准备

数据格式为 HDF5，结构如下：
```
data.h5
├── TFM/        # 低分辨率输入（TFM 成像结果）
│   ├── 0/      # 样本编号
│   ├── 1/
│   └── ...
└── hr/         # 高分辨率真值
    ├── 0/
    ├── 1/
    └── ...
```

数据生成参考：<https://github.com/cacdcaecawae/data-for-TFM-kwave>

### 测试

运行单元测试确保环境配置正确：
```bash
pytest
```

---

## 参考资源

- 项目规范：[AGENTS.md](AGENTS.md)
- 数据生成脚本：<https://github.com/cacdcaecawae/data-for-TFM-kwave>
- DeepWiki 学习平台：<https://deepwiki.org/>
- Python 环境搭建：[视频教程](https://www.bilibili.com/video/BV1bQ4y1n7sn/)
