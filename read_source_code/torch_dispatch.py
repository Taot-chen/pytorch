import torch

class PyTensor(torch.Tensor):
    __torch_dispatch__ = torch._C._disabled_torch_dispatch_impl
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError

if __name__ == "__main__":
    x = PyTensor(torch.randn(8))
    x * 2
