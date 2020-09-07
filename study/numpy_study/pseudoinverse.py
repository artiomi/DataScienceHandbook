import numpy as np
import numpy.linalg as nla

A = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]])
print("original:\n", A)
# calculate pseudoinverse
B = nla.pinv(A)
print("pseudoinverse:\n", B)

# step by step calculation
A = np.array([
    [0.1, 0.2], [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]])

U, s, V = nla.svd(A)
print("A shape:", A.shape)
print("U:\n", U)
print("s:\n", s)
print("V:\n", V)

# reciprocals of s
d = 1.0 / s
print("s recirpocal:\n", d)
# create m x n D matrix
D = np.zeros(A.shape)
# populate D with n x n diagonal matrix
D[:A.shape[1], :A.shape[1]] = np.diag(d)
# calculate pseudoinverse
B = V.T.dot(D.T).dot(U.T)
print("pseudoinverse: \n", B)
