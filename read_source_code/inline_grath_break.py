from typing import List
import torch

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")

    print(">>> FX graph:")
    gm.graph.print_tabular()

    print(f">>> Code:")
    print(gm.code)

    return gm.forward  # return a python callable

def baz(x):
    return -x if x > 0 else x - 1

def bar(x):
    return x * baz(x - 1)



@torch.compile(backend=my_compiler)
def foo(x):
    return x * bar(2 * x)

def test():
    x = torch.tensor([4])
    foo(x)

if __name__ == "__main__":
    test()
