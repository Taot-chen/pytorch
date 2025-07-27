import torch
from functorch.compile import aot_function, \
    make_boxed_func, ts_compile

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

def run_func(func, *inputs):
    res = func(*inputs)
    loss = res.sum()
    loss.backward()

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return make_boxed_func(fx_module.forward)


if __name__ == "__main__":
    a, b, c, d = [torch.randn(2, 4, requires_grad=True,
        device="cuda") for _ in range(4)]
    run_func(fn, a, b, c, d)

    aot_ts_nvfuser_fn = torch.compile(fn, backend="aot_ts_nvfuser")
    run_func(aot_ts_nvfuser_fn, a, b, c, d)

