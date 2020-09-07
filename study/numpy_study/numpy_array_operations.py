import numpy as np

arr_1d = np.array([11, 22, 33, 44])
print("np_array: ", arr_1d)
print("np_array type:", type(arr_1d))

arr_2d = np.array([
    [11, 22],
    [33, 44],
    [55, 66]])
print("first row:", arr_2d[0])
print("slice: [:, :-1]\n", arr_2d[:, :-1])
print("slice: [:, -1]\n", arr_2d[:, -1])

print("slice: [: 2, :]\n", arr_2d[: 2, :])
print("slice: [2:, :]\n", arr_2d[2:, :])

print('Rows: %d' % arr_2d.shape[0])
print('Cols: %d' % arr_2d.shape[1])

print("1D shape:", arr_1d.shape)
reshaped = arr_1d.reshape((2, 2, 1))
print("reshaped:\n", reshaped)
print(reshaped.shape)

sq_mat = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]])
print("sq_mat:\n", sq_mat)
print("upper triangle:\n", np.triu(sq_mat))
print("lower triangle:\n", np.tril(sq_mat))
diag_vect = np.diag(sq_mat)
print("diagonal vector:\n", diag_vect)
print("diagonal matrix:\n", np.diag(diag_vect))
print("identity matrix:\n", np.identity(3))

ort = np.array([
    [1, 0],
    [0, -1]
])

ort_inv = np.linalg.inv(ort)
print("ortagonal:\n", ort, ort.shape)
print("dot:\n", ort.dot(ort_inv))
print("ortagonal inv: \n", ort_inv)
print("ortag transpose:\n", ort.T)
# I = Q.dot(Q.T)
# print(I)
