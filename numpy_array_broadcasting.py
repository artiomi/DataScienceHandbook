import numpy as np

arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2, 3], [1, 2, 3]])
b = 3
vector = np.array([9, 8, 7])
arr_2d_4 = np.array([[1], [2], [3]])

arr_4d = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]])
new_ax = arr_1d[:, np.newaxis]
print("1d broadcast:", arr_1d + b)
print("2d broadcast:\n", arr_2d + b)
print("2d with vector broadcast:\n", arr_2d + vector, arr_2d.ndim, arr_2d.shape, vector.ndim, vector.shape)
print("4d broadcast:\n", arr_4d + arr_2d_4)
print("new axis: \n", new_ax , new_ax.shape, arr_2d.shape)
print(new_ax+vector)

iter  = arr_4d.flat
