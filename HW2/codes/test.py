import numpy as np
import math


a = np.array([[3, 4], [1, 6], [2, 2], [1, 0]])



def getMean(X):

	n = np.float64(X.shape[0])

	return X.sum(axis=0)/n


def getVar(X):
	# this function get the varaince of matrix X
	mean = np.ones(X.shape) * getMean(X)
	n = np.float64(np.shape(X)[0])

	var = float(1/n)*(np.power((X-mean),2))


def my_cov(X):
    """Returns the sample covariance matrix for X
    """
    mu = np.ones(np.shape(X)) * getMean(X)
    n = np.float64(np.shape(X)[0])
    return (1 / (n - 1)) * (X - mu).T.dot(X - mu)


def getCov(X):

	# this function get the Covariance of matrix X
	mean = np.ones(X.shape) * getMean(X)
	n = np.float64(X.shape[0])

	cov = (1/(n-1))* np.dot((X - mean),np.transpose(X-mean))
	
	return cov


print(a.T)

y = [0, 1]

for i in range(1):
	X_class = a[np.where(y == i)[0]]
	print(X_class)





