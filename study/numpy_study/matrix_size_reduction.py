import numpy as np
import numpy.linalg as nla
from sklearn.decomposition import TruncatedSVD

A = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])
print("initial matrix:\n", A)
# factorize
U, s, V = nla.svd(A)
print("U:", U.shape, "s:", s.shape, "V:", V.shape)
# create m x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
print("sigma with diagonal:\n", Sigma)
# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
print("sigma with n elements:\n", Sigma)

V = V[:n_elements, :]
print("reshaped V:", V.shape)
# reconstruct
B = U.dot(Sigma.dot(V))
print("reconstruct:\n", B)
# transform
T = U.dot(Sigma)
print("transform1:\n", T)
T = A.dot(V.T)
print("transform2:\n", T)

# SKI learn example

# create transform
svd = TruncatedSVD(n_components=2)
# fit transform
svd.fit(A)
# apply transform
result = svd.transform(A)
print("result:\n", result)
