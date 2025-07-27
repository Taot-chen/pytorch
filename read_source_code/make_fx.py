import torch
from torch.fx.experimental.proxy_tensor import make_fx

def f(x, y):
    return x + y


if __name__ == "__main__":
    x = torch.randn(8)
    y = torch.randn(8)
    g = make_fx(f)(x, y)
    print(g.code)
