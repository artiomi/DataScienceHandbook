import numpy as np
import numpy.linalg as nla
from sklearn.decomposition import PCA

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]])
# column means
M = np.mean(A.T, axis=1)
print("Mean:", M)
# center columns by subtracting column means
C = A - M
print("center columns:\n", C)
# calculate covariance matrix of centered matrix
V = np.cov(C.T)
print("covariance:\n", V)
# factorize covariance matrix
values, vectors = nla.eig(V)
print("eigencomposition vectors:\n", vectors)
print("eigencomposition values:\n", values)
# project data
P = vectors.T.dot(C.T)
print("project data:\n", P.T)

# principal component analysis with scikit-learn
# create the transform
pca = PCA(2)
# fit transform
pca.fit(A)
# access values and vectors
print("components:\n", pca.components_)
print("explained variance:\n", pca.explained_variance_)
print("mean:\n",pca.mean_)
# transform data
B = pca.transform(A)
print(B)
