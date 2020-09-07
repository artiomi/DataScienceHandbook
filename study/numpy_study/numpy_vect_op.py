import numpy as np

vect1 = np.array([1, 2, 3])
vect2 = np.array([6, 5, 4])
vect3 = np.array([[8], [9], [10]])
print("vector addition:\n", vect1 + vect2)
print("vect1", vect1.shape, vect3.shape)
print("sum:\n", vect3 + vect1)
print("substraction:\n", vect1 - vect3)

print("dot product v1 v2:", vect1.dot(vect2))
print("dot product v1 v3:", vect1.dot(vect3))
