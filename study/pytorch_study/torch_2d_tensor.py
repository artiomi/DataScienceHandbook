import torch as tc

a = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
A = tc.tensor(a)

print(A)
print("dimension: ", A.ndimension())
print("shape: ", A.shape)
print("size: ", A.size())
print("nr elements: ", A.numel())
print(A[1:3, 1])

X = tc.tensor([[1, 0], [0, 1]])
Y = tc.tensor([[2, 1], [1, 2]])
Z = X * Y
print("product X*Y:", Z)

A = tc.tensor([[0, 1, 1], [1, 0, 1]])
B = tc.tensor([[1, 1], [1, 1], [1, -1]])
C = tc.mm(A, B)
print("multiplication:", C)
