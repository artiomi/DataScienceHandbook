import numpy as np
import scipy.linalg as sla

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]])
# factorize
U, s, V = sla.svd(A)
print("A:\n", A, A.shape)
print("U:", U, U.shape)
print("s:", s, s.shape)
print("V:", V, V.shape)
diag = np.diag(s)
B = np.dot(U[:, :2] * s, V)
print("refactored:\n", B)

Sigma = np.zeros((A.shape[0], A.shape[1]))
print("sigma zero:\n", Sigma[:A.shape[1], :A.shape[1]])
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)
print("sigma with diag:\n", Sigma)
sigmaV = Sigma.dot(V)
B = U.dot(sigmaV)
print("refactored:\n", B)
print("sigma v :\n", sigmaV)
