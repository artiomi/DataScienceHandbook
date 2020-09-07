import numpy as np

A = np.array([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19]
    ],
    [
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29]
    ]])

B = np.array([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19]
    ],
    [
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29]
    ]])

print("A shape:", A.shape, A.ndim, "B shape:", B.shape, B.ndim)
ax0 = np.tensordot(A, B, axes=0)
ax1 = np.tensordot(A, B, axes=1)
ax2 = np.tensordot(A, B, axes=2)
ax3 = np.tensordot(A, B, axes=3)

print("axes 0:", ax0[-1, -1, -1], ax0.shape)
print("axes 1:", ax1, ax1.shape)
print("axes 2:", ax2, ax2.shape)
print("axes 3:", ax3, ax3.shape)


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [21, 22, 23],
    [24, 25, 26],
    [27, 28, 29]
])

print("A shape:", A.shape, "B shape:", B.shape)
ax0 = np.tensordot(A, B, axes=0)
ax1 = np.tensordot(A, B, axes=1)

print("axes 0:", ax0, ax0.shape)
print("axes 1:", ax1, ax1.shape)
