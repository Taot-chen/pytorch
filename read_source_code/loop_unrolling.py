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
def toy_example(x, n):
    for i in range(1, n+1):
        x = x * i
    return x

def test():
    x = torch.randn(10)
    toy_example(x, 4)

if __name__ == "__main__":
    test()
