import torch

x = torch.tensor(2., requires_grad=True)
y = x ** 2 + 2 * x + 1
print("x:", x)
print("y:", y)
der = y.backward()
print(x.grad)

u = torch.tensor(1., requires_grad=True)
v = torch.tensor(2., requires_grad=True)
f = u * v + u ** 2
f.backward()
print(u.grad)
print(v.grad)
print(f)