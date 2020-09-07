import numpy as np
import matplotlib
import numpy.linalg as nla

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = np.array([
    [0.05, 0.12],
    [0.18, 0.22],
    [0.31, 0.35],
    [0.42, 0.38],
    [0.5, 0.49]])
X, y = data[:, 0], data[:, 1]
print("X before reshape:\n", X)
print("y:\n", y)
X = X.reshape((len(X), 1))
print("X after reshape:\n", X)
###############################
# scatter plot
# matplotlib.pyplot.scatter(X, y)
# matplotlib.pyplot.show()

###########################
# linear least squares
b = nla.inv(X.T.dot(X)).dot(X.T).dot(y)
# predict using coefficients
yhat = X.dot(b)
print("b:", b)
print("yhat:", yhat)


#####################################
# QR decomposition

Q, R = nla.qr(X)
print("Q:\n", Q)
print("R:\n", R)
b = nla.inv(R).dot(Q.T).dot(y)
# predict using coefficients
yhat = X.dot(b)
print("b:", b)
print("yhat:", yhat)


########################
# pseudoinverse
b = nla.pinv(X).dot(y)
# predict using coefficients
yhat = X.dot(b)
print("b:", b)
print("yhat:", yhat)


##################
# least squares
# calculate coefficients
b, residuals, rank, s = nla.lstsq(X, y, rcond=None)
# predict using coefficients
yhat = X.dot(b)

print("b:", b)
print("yhat:", yhat)
# plot data and predictions
plt.scatter(X, y)
plt.plot(X, yhat, color='red')
plt.show()
