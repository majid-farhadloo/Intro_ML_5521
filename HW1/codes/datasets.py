from sklearn.datasets import load_boston, load_digits
import numpy as np



def percentileAssignment (percentage,X,y):
    percentile = np.percentile(y,percentage)
    final_target = np.where(y>=percentile,1,0);
    return X,final_target

def load_dataset(sk_learn_function):
    x_features, classes = sk_learn_function(return_X_y=True) 
    return x_features,classes


