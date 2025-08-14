import torch

# 检查PyTorch版本
print(f"PyTorch version: {torch.__version__}")

# 检查是否可以使用CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# 如果CUDA可用，打印CUDA设备的数量和名称
if cuda_available:
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available.")

