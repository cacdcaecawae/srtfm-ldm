from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class H5PairedDataset(Dataset):
    """
    HDF5 嵌套结构的配对数据集，读取 LR/HR 样本。
    
    预期的 HDF5 结构：
        file.h5
        ├── TFM/
        │   ├── 000001/data
        │   ├── 000002/data
        │   └── ...
        └── hr/
            ├── 000001/data
            ├── 000002/data
            └── ...
    """

    def __init__(
        self,
        h5_path: str,
        lr_key: str = "TFM",
        hr_key: str = "hr",
        lr_dataset_name: Optional[str] = None,
        hr_dataset_name: Optional[str] = None,
        transpose_lr: bool = False,
        transpose_hr: bool = False,
        use_tfm_channels: bool = False,  # 是否使用 I, X, Y 三通道
        coord_range: Optional[tuple] = None,  # X, Y 坐标的归一化范围 ((x_min, x_max), (y_min, y_max))
        augment: bool = False,  # 是否启用数据增强
        h_flip_prob: float = 0.5,  # 水平翻转概率
        translate_prob: float = 0.5,  # 平移概率
        max_translate_ratio: float = 0.05,  # 最大平移比例
    ):
        self.h5_path = h5_path
        self.lr_key = lr_key
        self.hr_key = hr_key
        self.lr_dataset_name = lr_dataset_name
        self.hr_dataset_name = hr_dataset_name
        self.transpose_lr = transpose_lr
        self.transpose_hr = transpose_hr
        self.use_tfm_channels = use_tfm_channels
        self.coord_range = coord_range if coord_range else ((-1.0, 1.0), (-1.0, 1.0))  # 默认 X, Y 都是 [-1, 1]
        
        # 数据增强参数
        self.augment = augment
        self.h_flip_prob = h_flip_prob
        self.translate_prob = translate_prob
        self.max_translate_ratio = max_translate_ratio

        self._file: Optional[h5py.File] = None
        self._sample_names: Optional[List[str]] = None
        self._lr_paths: Optional[List[str]] = None
        self._hr_paths: Optional[List[str]] = None

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

    def _init_structure(self) -> None:
        """初始化数据集结构，建立 LR/HR 样本映射"""
        if self._sample_names is not None:
            return

        f = self._open_file()
        
        if self.lr_key not in f or self.hr_key not in f:
            raise KeyError(f"HDF5 文件缺少 '{self.lr_key}' 或 '{self.hr_key}' 分组")

        lr_group = f[self.lr_key]
        hr_group = f[self.hr_key]

        # 找到共同的样本编号
        lr_names = set(lr_group.keys())
        hr_names = set(hr_group.keys())
        shared = sorted(lr_names & hr_names)
        
        if not shared:
            raise ValueError(f"LR 和 HR 分组下没有共同的样本编号")

        self._sample_names = shared
        self._lr_paths = []
        self._hr_paths = []

        # 构建每个样本的 dataset 路径
        for name in shared:
            self._lr_paths.append(self._get_dataset_path(lr_group[name], self.lr_dataset_name, name))
            self._hr_paths.append(self._get_dataset_path(hr_group[name], self.hr_dataset_name, name))

    def _get_dataset_path(self, node: h5py.Group, dataset_name: Optional[str], sample_name: str) -> str:
        """从分组节点中获取 dataset 路径"""
        if isinstance(node, h5py.Dataset):
            return node.name

        # 如果指定了 dataset 名称，直接使用
        if dataset_name:
            if dataset_name not in node:
                raise KeyError(f"样本 '{sample_name}' 下找不到 dataset '{dataset_name}'")
            return node[dataset_name].name

        # 否则自动查找唯一的 dataset
        datasets = [child.name for child in node.values() if isinstance(child, h5py.Dataset)]
        if len(datasets) == 1:
            return datasets[0]
        elif len(datasets) == 0:
            raise ValueError(f"样本 '{sample_name}' 下没有 dataset")
        else:
            raise ValueError(f"样本 '{sample_name}' 下有多个 dataset，请指定 dataset_name 参数")

    @staticmethod
    def _normalize(array: np.ndarray, transpose: bool = False) -> torch.Tensor:
        """将 numpy 数组归一化到 [-1, 1]"""
        # MATLAB 转置：MATLAB 列优先存储，Python 行优先
        # MATLAB (H, W) → HDF5 读为 (W, H)
        # MATLAB (H, W, C) → HDF5 读为 (C, W, H)
        if transpose:
            if array.ndim == 2:
                # 2D 灰度图：(W, H) → (H, W)
                array = array.T
            elif array.ndim == 3:
                # 3D 数组：假设是 (C, W, H)，交换 W 和 H
                # (C, W, H) → (C, H, W)
                array = np.transpose(array, (0, 2, 1))
        
        # 确保是 [C, H, W] 格式（添加通道维度）
        if array.ndim == 2:
            array = array[None, ...]  # (H, W) → (1, H, W)
        
        tensor = torch.from_numpy(array).float()
        
        # Min-max 归一化到 [0, 1]
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
        
        # 映射到 [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor

    @staticmethod
    def _normalize_intensity(array: np.ndarray) -> torch.Tensor:
        """归一化强度图像 (I)，使用自适应 min-max 到 [-1, 1]"""
        if array.ndim == 2:
            array = array[None, ...]  # (H, W) → (1, H, W)
        
        tensor = torch.from_numpy(array).float()
        
        # 自适应 min-max 归一化
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)  # → [0, 1]
        
        tensor = tensor * 2.0 - 1.0  # → [-1, 1]
        return tensor

    @staticmethod
    def _normalize_coords(array: np.ndarray, coord_range: tuple) -> torch.Tensor:
        """归一化坐标数据 (X/Y)，使用固定范围到 [-1, 1]"""
        if array.ndim == 2:
            array = array[None, ...]  # (H, W) → (1, H, W)
        
        tensor = torch.from_numpy(array).float()
        
        # 使用固定范围归一化
        min_val, max_val = coord_range
        tensor = (tensor - min_val) / (max_val - min_val)  # → [0, 1]
        tensor = tensor * 2.0 - 1.0  # → [-1, 1]
        
        return tensor
    
    def _apply_augmentation(
        self,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor
    ) -> tuple:
        """
        应用数据增强
        
        Args:
            lr_tensor: [C, H, W] - TFM模式下 C=3 (I,X,Y)，单通道模式 C=1
            hr_tensor: [1, H, W]
        
        Returns:
            增强后的 (lr_tensor, hr_tensor)
        """
        if not self.augment:
            return lr_tensor, hr_tensor
        
        if self.use_tfm_channels and lr_tensor.shape[0] == 3:
            # TFM 三通道模式：特殊处理
            return self._augment_tfm(lr_tensor, hr_tensor)
        else:
            # 单通道模式：标准处理
            return self._augment_standard(lr_tensor, hr_tensor)
    
    def _augment_standard(
        self,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor
    ) -> tuple:
        """标准单通道数据增强"""
        # 水平翻转
        if torch.rand(1).item() < self.h_flip_prob:
            lr_tensor = torch.flip(lr_tensor, dims=[-1])
            hr_tensor = torch.flip(hr_tensor, dims=[-1])
        
        # 平移
        if torch.rand(1).item() < self.translate_prob:
            lr_tensor, hr_tensor = self._apply_translation(lr_tensor, hr_tensor)
        
        return lr_tensor, hr_tensor
    
    def _augment_tfm(
        self,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor
    ) -> tuple:
        """TFM 三通道特殊增强"""
        I_channel = lr_tensor[0:1]  # [1, H, W]
        X_channel = lr_tensor[1:2]  # [1, H, W]
        Y_channel = lr_tensor[2:3]  # [1, H, W]
        
        # 水平翻转：X 坐标需要取反
        if torch.rand(1).item() < self.h_flip_prob:
            # 关键：先取反 X 坐标，再进行空间翻转
            X_channel = -X_channel
            
            I_channel = torch.flip(I_channel, dims=[-1])
            X_channel = torch.flip(X_channel, dims=[-1])
            Y_channel = torch.flip(Y_channel, dims=[-1])
            hr_tensor = torch.flip(hr_tensor, dims=[-1])
        
        # 平移：坐标值需要同步调整
        if torch.rand(1).item() < self.translate_prob:
            I_channel, X_channel, Y_channel, hr_tensor = self._apply_tfm_translation(
                I_channel, X_channel, Y_channel, hr_tensor
            )
        
        # 重新拼接
        lr_tensor = torch.cat([I_channel, X_channel, Y_channel], dim=0)
        
        return lr_tensor, hr_tensor
    
    def _apply_translation(
        self,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor
    ) -> tuple:
        """应用平移变换（使用边界复制填充）"""
        _, H, W = lr_tensor.shape
        
        # 随机平移量（像素）
        max_tx = int(W * self.max_translate_ratio)
        max_ty = int(H * self.max_translate_ratio)
        
        tx = torch.randint(-max_tx, max_tx + 1, (1,)).item()
        ty = torch.randint(-max_ty, max_ty + 1, (1,)).item()
        
        if tx == 0 and ty == 0:
            return lr_tensor, hr_tensor
        
        # 应用平移（全部用nearest保持原始像素值）
        lr_tensor = self._translate_tensor(lr_tensor, tx, ty, mode='nearest')
        hr_tensor = self._translate_tensor(hr_tensor, tx, ty, mode='nearest')
        
        return lr_tensor, hr_tensor
    
    def _apply_tfm_translation(
        self,
        I_channel: torch.Tensor,
        X_channel: torch.Tensor,
        Y_channel: torch.Tensor,
        hr_tensor: torch.Tensor
    ) -> tuple:
        """TFM 三通道平移（强度做空间平移,坐标直接平移数值）"""
        _, H, W = I_channel.shape
        
        # 随机平移量（像素）
        max_tx = int(W * self.max_translate_ratio)
        max_ty = int(H * self.max_translate_ratio)
        
        tx = torch.randint(-max_tx, max_tx + 1, (1,)).item()
        ty = torch.randint(-max_ty, max_ty + 1, (1,)).item()
        
        if tx == 0 and ty == 0:
            return I_channel, X_channel, Y_channel, hr_tensor
        
        # 1. 对强度通道(I)和HR图像应用空间平移（全部用nearest保持原始像素值）
        I_channel = self._translate_tensor(I_channel, tx, ty, mode='nearest')
        hr_tensor = self._translate_tensor(hr_tensor, tx, ty, mode='nearest')
        
        # 2. X/Y坐标通道:不做空间平移,只调整坐标值
        # 
        # 物理模型:
        # - 图像实际物理大小: 5mm × 5mm
        # - 图像像素: H×W (例如 101×101)
        # - 像素间隔数: (H-1) × (W-1) = 100 × 100
        # - 每个间隔物理大小: 5/(H-1) mm × 5/(W-1) mm
        # - X坐标总范围: x_max - x_min (例如 0.021mm)
        # - Y坐标总范围: y_max - y_min (例如 0.035mm)
        # - X/Y已经归一化到 [-1, 1]
        #
        # 平移1个像素时,归一化坐标的变化:
        # - 物理平移: 5/(W-1) mm
        # - 占总范围的比例: [5/(W-1)] / (x_max-x_min)
        # - 归一化坐标变化: 比例 × 2.0 (因为归一化范围是2,从-1到1)
        # - 所以: dx_norm = [5/(W-1)] / (x_max-x_min) × 2.0
        
        (x_min, x_max), (y_min, y_max) = self.coord_range
        
        # 图像实际物理大小 (单位与坐标范围一致)
        physical_size = 0.008  # 8mm = 0.008m (假设坐标单位是m)
        
        # 每像素的归一化坐标变化量
        # 归一化范围是 2.0 (从-1到1)
        coord_per_pixel_x = (physical_size / (W - 1)) / (x_max - x_min) * 2.0
        coord_per_pixel_y = (physical_size / (H - 1)) / (y_max - y_min) * 2.0
        
        # 坐标偏移量(归一化坐标)
        dx_coord = tx * coord_per_pixel_x
        dy_coord = ty * coord_per_pixel_y
        
        # 直接给所有坐标值加上偏移量(不做空间平移)
        # 平移后坐标范围会略微超出[-1, 1],这是正常的
        X_channel = X_channel + dx_coord
        Y_channel = Y_channel + dy_coord
        
        # 不裁剪坐标范围,允许平移后范围改变
        
        return I_channel, X_channel, Y_channel, hr_tensor
    
    @staticmethod
    def _translate_tensor(
        tensor: torch.Tensor,
        tx: int,
        ty: int,
        mode: str = 'bilinear'
    ) -> torch.Tensor:
        """
        平移张量（使用边界复制填充）
        
        Args:
            tensor: [C, H, W]
            tx: 水平平移（像素，正数向右）
            ty: 垂直平移（像素，正数向下）
            mode: 插值模式，'bilinear'或'nearest'
        
        Returns:
            平移后的张量
        """
        C, H, W = tensor.shape
        
        # 构建仿射变换矩阵
        # PyTorch 的 grid_sample 使用归一化坐标 [-1, 1]
        # 平移量需要转换为归一化坐标
        theta = torch.tensor([
            [1, 0, 2.0 * tx / W],
            [0, 1, 2.0 * ty / H]
        ], dtype=tensor.dtype, device=tensor.device).unsqueeze(0)  # [1, 2, 3]
        
        # 生成采样网格
        grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)
        
        # 应用变换（使用边界复制填充）
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]
        tensor = F.grid_sample(
            tensor,
            grid,
            mode=mode,  # 使用指定的插值模式
            padding_mode='border',  # 关键：使用边界复制而非零填充
            align_corners=False
        )
        tensor = tensor.squeeze(0)  # [C, H, W]
        
        return tensor

    def __len__(self) -> int:
        self._init_structure()
        return len(self._sample_names)

    def __getitem__(self, idx: int):
        self._init_structure()
        f = self._open_file()

        sample_name = self._sample_names[idx]
        
        # 处理 LR (TFM) 数据
        if self.use_tfm_channels:
            # 读取 I, X, Y 三个数据集
            lr_group = f[self._lr_paths[idx]].parent  # 获取父分组
            
            # 读取三个通道（支持 'I' 或 'intensity'）
            I_array = None
            if 'I' in lr_group:
                I_array = np.asarray(lr_group['I'])
            elif 'intensity' in lr_group:
                I_array = np.asarray(lr_group['intensity'])
            else:
                I_array = np.asarray(f[self._lr_paths[idx]])
            
            X_array = np.asarray(lr_group['X']) if 'X' in lr_group else None
            Y_array = np.asarray(lr_group['Y']) if 'Y' in lr_group else None
            
            # 转置处理
            if self.transpose_lr:
                I_array = I_array.T if I_array.ndim == 2 else np.transpose(I_array, (0, 2, 1))
                if X_array is not None:
                    X_array = X_array.T if X_array.ndim == 2 else np.transpose(X_array, (0, 2, 1))
                if Y_array is not None:
                    Y_array = Y_array.T if Y_array.ndim == 2 else np.transpose(Y_array, (0, 2, 1))
            
            # 归一化 I（强度）：自适应 min-max
            I_tensor = self._normalize_intensity(I_array)
            
            # 归一化 X, Y（坐标）：固定范围
            if X_array is not None and Y_array is not None:
                X_tensor = self._normalize_coords(X_array, self.coord_range[0])
                Y_tensor = self._normalize_coords(Y_array, self.coord_range[1])
                # 拼接成 [3, H, W]
                lr_tensor = torch.cat([I_tensor, X_tensor, Y_tensor], dim=0)
            else:
                # 如果没有 X, Y，只用 I
                lr_tensor = I_tensor
        else:
            # 原来的单通道模式
            lr_array = np.asarray(f[self._lr_paths[idx]])
            lr_tensor = self._normalize(lr_array, transpose=self.transpose_lr)
        
        # 处理 HR 数据（保持原样）
        hr_array = np.asarray(f[self._hr_paths[idx]])
        hr_tensor = self._normalize(hr_array, transpose=self.transpose_hr)

        # 应用数据增强
        lr_tensor, hr_tensor = self._apply_augmentation(lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor, sample_name

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


def get_h5_dataloader(
    h5_path: str,
    batch_size: int,
    lr_key: str = "TFM",
    hr_key: str = "hr",
    lr_dataset_name: Optional[str] = None,
    hr_dataset_name: Optional[str] = None,
    transpose_lr: bool = False,
    transpose_hr: bool = False,
    use_tfm_channels: bool = False,
    coord_range: Optional[tuple] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = False,  # 是否启用数据增强
    h_flip_prob: float = 0.5,  # 水平翻转概率
    translate_prob: float = 0.5,  # 平移概率
    max_translate_ratio: float = 0.05,  # 最大平移比例
) -> DataLoader:
    """创建 HDF5 数据集的 DataLoader
    
    Args:
        use_tfm_channels: 是否使用 TFM 的 I, X, Y 三通道模式
        coord_range: X, Y 坐标的归一化范围，格式为 ((x_min, x_max), (y_min, y_max))
        augment: 是否启用数据增强
        h_flip_prob: 水平翻转概率
        translate_prob: 平移概率
        max_translate_ratio: 最大平移比例（相对于图像大小）
    """
    dataset = H5PairedDataset(
        h5_path=h5_path,
        lr_key=lr_key,
        hr_key=hr_key,
        lr_dataset_name=lr_dataset_name,
        hr_dataset_name=hr_dataset_name,
        transpose_lr=transpose_lr,
        transpose_hr=transpose_hr,
        use_tfm_channels=use_tfm_channels,
        coord_range=coord_range,
        augment=augment,
        h_flip_prob=h_flip_prob,
        translate_prob=translate_prob,
        max_translate_ratio=max_translate_ratio,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
