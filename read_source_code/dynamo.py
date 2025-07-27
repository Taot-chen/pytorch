from typing import List
import torch

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")

    print(">>> FX graph:")
    gm.graph.print_tabular()

    print(f">>> Code:")
    print(gm.code)

    return gm.forward  # return a python callable


@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x

def foo_1(x, y):
    return x * y + x - y


if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    
    print("\n使用装饰器的方式调用:")
    foo(a, b)
    
    # 另一种调用方式：
    print("\n直接调用 torch.compile():")
    complied_code = torch.compile(foo_1, backend=my_compiler)
    print("compiled code: ", complied_code)
    ret = complied_code(a, b)
    print("ret: ", ret)
