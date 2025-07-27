import torch
from torch.fx.experimental.proxy_tensor import make_fx

def f(x, y):
    return x + y


if __name__ == "__main__":
    x = torch.randn(8)
    g = make_fx(f)(x, x)
    print(g.code)

    y = torch.randn(1, requires_grad=True)
    ret = torch.autograd.grad(f(y, y), (y, y))
    print(ret)
    
    y = torch.randn(1, requires_grad=True)
    z = torch.randn(1, requires_grad=True)
    ret = torch.autograd.grad(f(y, z), (y, z))
    print(ret)

