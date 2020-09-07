import numpy as np

# Mean

v = np.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]])
col_mean = np.mean(v, axis=0)
row_mean = np.mean(v, axis=1)
print("col mean:", col_mean)
print("row mean:", row_mean)

# calculate variance
col_variance = np.var(v, ddof=1, axis=0)
row_variance = np.var(v, ddof=1, axis=1)
print("col_variance:", col_variance)
print("row_variance:", row_variance)

# deviation
# column standard deviations
col_std = np.std(v, ddof=1, axis=0)
print("column deviation:", col_std)
# row standard deviations
row_std = np.std(v, ddof=1, axis=1)
print("row deviation:", row_std)

# covariance
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 7, 8, 9])

Sigma = np.cov(x, y)[0, 1]
print("covariance:\n", Sigma)
# calculate correlation
corr = np.corrcoef(x, y)
print("correlation:", corr)

X = np.array([
    [1, 5, 8],
    [3, 5, 11],
    [2, 4, 9],
    [3, 6, 10],
    [1, 5, 10]])
# calculate covariance matrix
Sigma = np.cov(X.T)
print("covariance matrix:\n", Sigma)
