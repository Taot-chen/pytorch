import torch
import torch._dynamo as dynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("ohuo")
    if b.sum() < 0:
        b = b * -1
    return x * b

ret = dynamo.explain(
    toy_example, torch.randn(10), torch.randn(10)
)

print("ret: \n", ret)
