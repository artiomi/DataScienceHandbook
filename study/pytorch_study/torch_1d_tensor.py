import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt

sample = torch.tensor([3, 6, 7, 2, 4, 9, 11, 12])
a = torch.tensor([i for i in range(1, 17)])
print(a)
print(a.dtype)
print(a.type())
print("size:{}".format(a.size()))
print("dimension:{}".format(a.ndimension()))

a_view = a.view(2, -1)
print(a_view)
print("dimension:{}".format(a_view.ndimension()))

np_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
torch_tensor = torch.from_numpy(np_array)
back_to_numpy = torch_tensor.numpy()
print("before update")
print("numpy", np_array)
print("tensor", torch_tensor)
print("back_to_numpy", back_to_numpy)

back_to_numpy[3] = 13
np_array[0] = 10
torch_tensor[2] = 12

print("after update")
print("numpy", np_array)
print("tensor", torch_tensor)
print("back_to_numpy", back_to_numpy)

print("operations")
u = torch.tensor([1, 2])
v = torch.tensor([3, 4])
z = torch.tensor(data=[0], dtype=torch.int64)
torch.dot(u, v, out=z)
print(z)

with_constant = sample + 3
print(with_constant)

test_function = torch.tensor([0, np.pi / 2, np.pi])
print(test_function)
calc_sin = torch.sin(test_function)
print(calc_sin)

ls = torch.linspace(0, 2 * np.pi, steps=100)
sn = torch.sin(ls)

plt.plot(ls.numpy(), sn.numpy())
