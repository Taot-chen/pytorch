import torch
from torch.fx import symbolic_trace
from torch.fx import subgraph_rewriter


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

if __name__ == "__main__":
    module = MyModule()
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)


    pattern = lambda x, y: x + y
    replacement = lambda x, y: x * y
    subgraph_rewriter.replace_pattern(symbolic_traced, pattern, replacement)
    print(symbolic_traced.graph)
    print(symbolic_traced.code)
    
    x = torch.randn((3, 4))
    print("x: ", x)
    dy =torch.randn((3, 5))
    print("dy: ", dy)

    y = symbolic_traced(x)
    print("y: ", y)
    y.backward(dy)
    print("y: ", y)
    
