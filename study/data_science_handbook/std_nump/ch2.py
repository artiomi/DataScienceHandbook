import timeit

import numpy as np
from scipy import special

np.random.seed(0)  # seed for reproducibility
x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")

print("x3:", x3[0:2, :, 0])

print("x2:\n", x2)
print("x2 inversed:\n", x2[::-1])
x2_sub_copy = x2[:2, :2].copy()
print("x2_sub_copy:\n", x2_sub_copy)

grid = np.arange(1, 9).reshape((2, 4))
print("reshape:\n", grid)

x = np.array([1, 2, 3])
print(x, x.shape)
nax_row = x[np.newaxis, :]
print("new axis row:\n", nax_row, nax_row.shape)
nax_col = x[:, np.newaxis]
print(np.newaxis)
print("new axis:\n", nax_col, nax_col.shape)

# print("x3: \n", x3)
# x3_new_ax = x3[:, :, :, np.newaxis]
# print("x3 new axis:\n", x3_new_ax, x3_new_ax.shape)

#####################
# array concatenation
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
concat = np.concatenate([grid, grid], axis=1)
print("concat1:\n", concat)

# concatenation
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
x1[0] = 12
print("x:", x)

# unfuncs arithmetic operation
x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
abs = np.abs(x)
print("abs:", abs, x.dtype)

##trigonometric functions
theta = np.linspace(0, np.pi, 3)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

# exponents and logarithms
x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))

x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))

# scipy special functions
x = np.array([1, 5, 10])
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))

# aggregation
reduced = np.add.reduce(x)
at = np.add.at(x, [0, 1], 1)
print("reduced:", reduced)
print("at:", x)

# outer products
x = np.arange(1, 6)
print("x:\n", x)
outer = np.multiply.outer(x, x)
print("outer:", outer)


