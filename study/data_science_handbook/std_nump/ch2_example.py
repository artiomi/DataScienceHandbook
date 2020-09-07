import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

ROOT = '/work/workspaces/pycharm/training_data_science/'
data = pd.read_csv(ROOT + 'data/Seattle2014.csv')
sample_data = np.array(data[:10])
rainfall = data['PRCP'].values
inches = rainfall / 254  # 1/10mm -> inches
snow = data['SNOW'].values

# plt.hist(inches, 40)
# plt.show()
# plt.hist(snow, 40)
# plt.show()

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
print(np.count_nonzero(x < 6))
print(np.any(x > 8))
print(np.all(x > 8))
print("Number days without rain: ", np.sum(inches == 0))
print("Number days with rain: ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.1 inches :", np.sum((inches > 0) &
                                               (inches < 0.2)))
# masking operation
less = x[x < 5]
print("less:", less)

# statistics of rain
snowy = (snow > 0)
rainy = (inches > 0)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)

winter = (np.arange(365) < 59) | (np.arange(365) > 334)
print("Median precip on rainy days in 2014:", np.median(inches[rainy]))
print("Median precip in summer 2014:", np.median(inches[summer & rainy]))
print("Maximum precip in summer 2014:", np.max(inches[summer]))
print("Median precip on non summer rainy days:", np.median(inches[rainy & ~ summer]))

print("Median snow:", np.median(snow[snowy]))
print("Median snow in winter:", np.median(snow[winter & snowy]))
print("Max snow in winter:", np.where(snowy))
print("Max snow in winter:", snow[snowy])

rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
ind = np.array([[1, 2], [3, 4]])
print("fancy index:\n", x[ind])
X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print("multi level:\n", X[row, col])

# ex 2
mean = [0, 0]
cov = [[1, 2], [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
print("X shape:", X.shape)

indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]

# plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
# plt.scatter(selection[:, 0], selection[:, 1],  facecolor='green')
# plt.show()

##############
sample = np.random.randint(1, 100, 10)
print(sample)
i = np.array([2, 1, 8, 4])
sample[i] += 10
print(sample)

i = [2, 3, 3, 4, 4, 4]
x = np.zeros(10)
np.add.at(x, i, 1)
print("test at:", x)

np.random.seed(42)
x = np.random.randn(100)
print("x:", x)
# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
print("bins:", bins)
counts = np.zeros_like(bins)
print("counts:", counts.size)
# find the appropriate bin for each x
i = np.searchsorted(bins, x)
print("i:", i, i.size)
# add 1 to each of these bins
np.subtract.at(counts, i, 1)
print("counts:", counts)


# plt.plot(bins, counts, linestyle='dotted')
# plt.hist(x, bins, histtype='step')
# plt.show()


