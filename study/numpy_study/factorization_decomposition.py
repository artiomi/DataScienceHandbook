import numpy as np
import scipy.linalg as la

# LPU
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

P, L, U = la.lu(A)
print("P:\n", P)
print("L:\n", L)
print("U:\n", U)
B = P @ L @ U
print("Reconstructed:\n", B)

# QR

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]])
# factorize
Q, R, P = la.qr(A, mode='full', pivoting=True)
print("Q:\n", Q)
print("R:\n", R)
print("P:\n", P)
# reconstruct
B = Q @ R @ P
print("reconstructed:\n", B)
# choesky
A = np.array([
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2]])

L = la.cholesky(A,lower=True)
print("L:\n", L)
print("L.T:\n", L.T)
# reconstruct
B = L @ L.T
print("reconstruct:\n", B)
