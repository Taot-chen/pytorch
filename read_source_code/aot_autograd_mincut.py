import torch
from functorch.compile import aot_function, make_boxed_func, ts_compile
from functorch.compile import min_cut_rematerialization_partition

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

    aot_mincut_fn = aot_function(
        fn,
        fw_compiler=compiler_fn,
        bw_compiler=compiler_fn,
        decompositions=None
    )
    run_func(aot_mincut_fn, a, b, c, d)
