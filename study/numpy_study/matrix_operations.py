import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse

org = np.array([
    [1, 2],
    [3, 4]
])

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

print("original:\n", org, org.ndim)
print("transpose:\n", org.T)
print("inverse:\n", la.inv(org))
print("trace:", np.trace(org))
print("determinant:", la.det(org), la.det(A))
print("inverse:\n", la.inv(A))
M2 = np.array([
    [1, 2],
    [3, 4]])
print(M2)
mr2 = la.matrix_rank(M2)
print(mr2)
# CSR
dense_arr = np.array([
    [1, 3, 5, 1, 8, 11],
    [23, 0, 2, 67, 0, 1],
    [67, 43, 0, 2, 11, 9]])
csr = sparse.csr_matrix(dense_arr)
print("CSR:\n", csr, "\nshape:", csr.shape, "data type:", csr.dtype)
print("todense:\n", csr.todense(), "shape:", dense_arr.shape)

non_zero = np.count_nonzero(dense_arr)
print("non zero:", non_zero, "size:", dense_arr.size)
sparcity = 1.0 - non_zero / dense_arr.size
print("sparcity:", sparcity)

# BSR matrix
bsr = sparse.bsr_matrix(dense_arr)
sparcity = 1.0 - bsr.count_nonzero() / bsr.nnz
print("BSR:\n", bsr)
print("BSR sparcity:", sparcity)
print("bsr to array:", bsr.get_shape())

row = np.array([0, 0, 1, 2, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6, 5])
matrix = sparse.bsr_matrix((data, (row, col)), shape=(3, 3))
print(matrix.toarray())
print(np.max(matrix.data))

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
sparse2 = sparse.bsr_matrix((data, indices, indptr), shape=(6, 6))
print("BSR:\n", sparse2.toarray())

# COO matrix
row = np.array([0, 0, 1, 3, 1, 0, 0])
col = np.array([0, 2, 1, 3, 1, 0, 0])
data = np.array([1, 1, 1, 1, 1, 1, 1])
coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
# Duplicate indices are maintained until implicitly or explicitly summed
print("COO max:", np.max(coo.data))
print("COO Array:\n", coo.toarray())

# CSR
docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
indptr = [0]
indices = []
data = []
vocabulary = {}
for d in docs:
    for term in d:
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))

print("data:", data, "indptr:", indptr, "indeces:", indices)
csr = sparse.csr_matrix((data, indices, indptr), dtype=int, shape=(2, 3)).toarray()
print("csr:\n", csr, csr.shape)

# DIA
n = 5
ones = np.ones(n)
data = np.array([3 * ones, 2 * ones, ones])
offsets = np.array([-1, 0, 1])
dia_mat = sparse.dia_matrix((data, offsets), shape=(n, n)).toarray()
print("Ones:\n", ones)
print("Dia matrix:\n", dia_mat)
print("data:\n", data)

# DOC
S = sparse.dok_matrix((5, 5), dtype=np.float32)
print("DOC before:\n", S.toarray())
for i in range(5):
    for j in range(5):
        S[i, j] = i + j

print("DOC after:\n", S.toarray())
