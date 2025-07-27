import dis

def hello():
    print("Hello, world!")

for k in ["co_names", "co_varnames", "co_consts"]:
    print(k, getattr(hello.__code__, k))
print(dis.dis(hello))

