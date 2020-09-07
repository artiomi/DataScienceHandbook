from math import inf

import numpy as np
import numpy.linalg as linalg

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[1, 2], [3, 4], [5, 6]])

print("arr1:\n", arr1)
print("arr2:\n", arr2)
# print("rotate arr:\n", np.rot90(arr2))
print("flip None: \n", np.flip(arr2))
print("flip 1: \n", np.flip(arr2, axis=1))
print("flip 0: \n", np.flip(arr2, axis=0))

print("norm None:", linalg.norm(arr1))
print("norm fro:", linalg.norm(arr1, ord='fro'))
print("norm nuc:", linalg.norm(arr1, ord='nuc'))
print("norm inf:", linalg.norm(arr1, ord=inf))
print("norm -inf:", linalg.norm(arr1, ord=-inf))
# print("norm 0:", linalg.norm(arr1, ord=0))
print("norm 1:", linalg.norm(arr1, ord=1))
print("norm -1:", linalg.norm(arr1, ord=-1))
print("norm 2:", linalg.norm(arr1, ord=2))
print("norm -2:", linalg.norm(arr1, ord=-2))

print("sign:\n", np.sign(arr1))
print("dot:\n", arr1.dot(arr2))
print("dot alternative:\n", arr1 @ arr2)
