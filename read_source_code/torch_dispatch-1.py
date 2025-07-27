import torch
from torch.utils._python_dispatch import TorchDispatchMode

x = torch.randn(8)
with TorchDispatchMode():
    x * 2
