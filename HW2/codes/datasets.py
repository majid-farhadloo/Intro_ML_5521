from sklearn.datasets import load_boston, load_digits
import numpy as np



def percentileAssignment (percentage,X,y):
    percentile = np.percentile(y,percentage)
    final_target = np.where(y>=percentile,1,0);
    return X,final_target

def load_dataset(sk_learn_function):
    x_features, classes = sk_learn_function(return_X_y=True) 
    return x_features,classes


def add_noise(X, mu=0, sigma=1):
    """Adds gaussian noise to the given matrix X
    """
    G = np.random.normal(mu, sigma, X.size).reshape(X.shape)
    return X + G


def prepare_digits(want_noise=True):
    X, y = load_dataset(load_digits)
    

    # Add some gaussian noise to avoid singular covariance matrices
    if want_noise:
        X = add_noise(X, sigma=0.001)

    return X, y