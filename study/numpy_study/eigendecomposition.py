import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

values, vectors = np.linalg.eig(A)
print("matrix:\n", A)
print("values:\n", values)
print("vectors:\n", vectors)

B = A.dot(vectors[:, 0])
print("matrix with eigenvector:\n", B)
C = vectors[:, 0] * values[0]
print("values with vectors:\n", C)

# create matrix from eigenvectors
Q = vectors
# create inverse of eigenvectors matrix
R = np.linalg.inv(Q)
# create diagonal matrix from eigenvalues
L = np.diag(values)
# reconstruct the original matrix
reconstructed = Q.dot(L).dot(R)
print("reconstructed:\n", reconstructed)
print("inversed vectors:", R)
print("diagonal matrix:\n", L)
