import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

#
# def selection_sort(x):
#     count = 0
#     for i in range(len(x)):
#         swap = i + np.argmin(x[i:])
#         (x[i], x[swap]) = (x[swap], x[i])
#         count += 1
#     print("selection sort iterations:", count)
#     return x
#
#
# x = np.array([2, 1, 4, 3, 5])
# selection_sort(x)
#
#
# def bogosort(x):
#     count = 0
#     while np.any(x[:-1] > x[1:]):
#         np.random.shuffle(x)
#         count += 1
#     print("bogosort iterations count:", count)
#     return x
#
#
# x = np.array([2, 1, 4, 3, 5])
# bogosort(x)
# print("sorted x:", x)


x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print("indices:", i)

# K nearest neighbour

X = np.random.rand(10, 2)
print(X.shape)
print(X[:, np.newaxis, :].shape)
print(X[np.newaxis, :, :].shape)
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
print("dist sq:", dist_sq.shape)
print("dist sq diagonal:", dist_sq.diagonal())
nearest = np.argsort(dist_sq, axis=1)
print("nearest:\n", nearest)
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
# print("nearest partition:\n", nearest_partition)

plt.scatter(X[:, 0], X[:, 1], s=100)
# draw lines from each point to its two nearest neighbors
K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K + 1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')
# plt.scatter(X[:, 0], X[:, 1], s=100)
plt.show()
