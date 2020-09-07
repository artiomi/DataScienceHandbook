import numpy as np

arr = np.array([[1, 2, 3, 4], [11, 12, 13, 14]], dtype=np.int16)

print("array:", arr)
print("data type", arr.dtype)
print("shape:", arr.shape)

empty_arr = np.empty([3, 2])
print("array:", empty_arr)
print("data type", empty_arr.dtype)
print("shape:", empty_arr.shape)

empty_like = np.empty_like(arr)
print("array:", empty_like)
print("data type", empty_like.dtype)
print("shape:", empty_like.shape)

a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])
vs_arr = np.vstack((a1, a2))
print("vstack: ", vs_arr)
print("vstack shape: ", vs_arr.shape)

hs_arr = np.hstack((a1, a2))
print("hstack: ", hs_arr)
print("hstack shape: ", hs_arr.shape)

np_eye = np.eye(3)
print("eye:", np_eye)
np_full = np.full([3, 3], [00, 11, 22])
print("full:", np_full)
print("full info:", np_full.shape, np_full.ndim)

a1 = np.array([[[1, 2, 3], [4, 5, 6]]])
a2 = np.array([[[11, 12, 13], [14, 15, 16]]])
print("a1:", a1.ndim, a1.shape)
print("a2:", a2.ndim, a2.shape, a2.T)
print("concat on 0:", np.concatenate((a1, a2), axis=0))
print("concat on 1:", np.concatenate((a1, a2), axis=1))
print("concat on 2:", np.concatenate((a1, a2), axis=2))

np_block = np.block([a1, a2])
print("np_block:", np_block)

A = np.eye(2) * 2
B = np.eye(3) * 3
zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
np_block = np.block([
    [A, zeros],
    [ones, B]
])
print("A:\n", A)
print("B:\n", B)
print("zeros:\n", zeros)
print("ones:\n", ones)
print("np_block:\n", np_block)

np_rand_array = [np.random.randn(3, 4) for _ in range(5)]
np_stack_ax0 = np.stack(np_rand_array, axis=0)
np_stack_ax1 = np.stack(np_rand_array, axis=1)
np_stack_ax2 = np.stack(np_rand_array, axis=2)

print("np_rand_array:\n", np_rand_array)
# print("np_stack0:\n", np_stack_ax0.shape, np_stack_ax0)
print("np_stack1:\n", np_stack_ax1.shape, np_stack_ax1)
print("np_stack2:\n", np_stack_ax2.shape, np_stack_ax2)

col_stack = np.column_stack((a1, a2))
print("col_stack:\n", col_stack)
