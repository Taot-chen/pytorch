from torch.fx import symbolic_trace

def f(x, y):
    return x + y

if __name__ == "__main__":
    h = symbolic_trace(f)
    print(h.code)
