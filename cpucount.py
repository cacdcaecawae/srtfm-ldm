import torch
print("PyTorch:", torch.__version__)                 # e.g. 2.3.1+cu121
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA (build):", torch.version.cuda)   # e.g. '12.1'；若为 None 说明是 CPU 版
print("cuDNN:", torch.backends.cudnn.version())      # e.g. 8905
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))  # 算力
