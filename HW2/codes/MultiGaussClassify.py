import numpy as np
import numpy.linalg as la
import math

def getMean(X):

	# this function get the mean of each colmn
 
	n = np.float64(X.shape[0]) # get how many features X has. (dimension)

	return X.sum(axis=0)/n


def getCov(X):
# this function get the Covariance of matrix X
    # this function get the Covariance of matrix X
    mean = np.ones(X.shape) * getMean(X)
    n = np.float64(np.shape(X)[0])

    return (1 / (n - 1)) * (X - mean).T.dot(X - mean)

    
	


def getVar(X):
	# this function get the varaince of matrix X
	mean = np.ones(X.shape) * getMean(X)
	n = np.float64(X.shape[0])

	var = (1/n)*(np.power((X-mean),2))
 	



class QuadraticGaussianDiscriminant:
    def __init__(self,name, X,n):

        self.name  = name
        S = np.identity(X.shape[1])
        means = np.zeros(X.shape[1])

        # Now properly initalize S and means
        S = getCov(X)
        means = getMean(X)

        # Estimate the prior from our data
        prior = X.shape[0] / np.float64(n)

        # Calculate the inverse and determinant of S for later use
        Sinv = la.inv(S)
        Sdet = la.det(S)

        self.W_i = (-1/2.) * Sinv
        self.w_i = np.dot(Sinv,means)
        self.w_i0 = (-1/2.) * (
            means.T.dot(
                Sinv.dot(
                    means))) - (1/2.) * np.log(
                        Sdet) + np.log(prior)

    def discriminant(self, X):

        return X.T.dot(
            self.W_i.dot(
                X)) + self.w_i.T.dot(
                    X) + self.w_i0



class LinearGaussianDiscriminant:
    
    def __init__(self, name, X, n, S):
        # Keep our class label
        self.name = name

        # Initialize means, S is provided
        means = my_mean(X)

        # Estimate the prior from our data
        prior = X.shape[0] / np.float64(n)

        # Invert S for later use
        Sinv = la.inv(S)

        # Linear discriminant
        self.w_i = Sinv.dot(means)
        self.w_i0 = (-1/2.) * (
            means.T.dot(Sinv.dot(
                means))) + np.log(prior)

    def discriminant(self, X):
        
        return self.w_i.T.dot(X) + self.w_i0

	    
class MultiGaussClassify:

    def __init__(self, k, d, diag=False):

    	cov = np.identity(d)
    	means = np.zeros(d)
    	prior = np.ones(k)*(1/float(k))
	
    def fit(self, X, y, diag=False):
        # Create new discriminants for each fit
        self.classes = []

        # Initalize the number of observations
        n = X.shape[0]

        
        
        # Get the classes from the data
        classes = np.unique(y)
        ncols = X.shape[1]


        if diag:
            S_common = getCov(X)

        # Create and initalize one discriminant per class
        for c in classes:
            # Separate out the data for this class
            X_class = X[np.where(y == c)[0]]
            X_class.reshape(X_class.shape[0], ncols)

            # determine if we want to go with LinearGaussianDiscriminant 
            if diag:
                discriminant = LinearGaussianDiscriminant(
                    c, X_class, n, S_common)
            else:
            
                discriminant = QuadraticGaussianDiscriminant(c, X_class, n)

            self.classes.append(discriminant)

    def predict(self, X):
        
        ypred = []

        for i in np.arange(X.shape[0]):
            scores = []
            # Evaluate the discrimant for each class
            for cls in self.classes:
                s = cls.discriminant(X[i].T)
                scores.append(s)
            # Find the highest score
            i = np.argmax(scores)
            # Return the label for the class with the highest score
            ypred.append(self.classes[i].name)

        return np.array(ypred)