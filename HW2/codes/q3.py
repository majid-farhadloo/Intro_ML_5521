from sklearn.datasets import load_boston, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from MultiGaussClassify import MultiGaussClassify
import numpy as np

from datasets import (
    percentileAssignment,
    load_dataset, prepare_digits
    )

from my_cross_val import my_cross_val
from utils import (
    report,
    wrapper_args
    )

import sys



import warnings
warnings.filterwarnings('ignore', '.*')


def q3(argv=None):

    dataset, method_name, k, latex = wrapper_args(
        argv, 'q3', ['Boston50', 'Boston75', 'Digits'])

    x_boston, y_boston = load_dataset(load_boston)
    x_digits, y_digits = prepare_digits(True)
    x_boston_50, y_boston_50 = percentileAssignment(50,x_boston,y_boston)
    x_boston_75, y_boston_75 = percentileAssignment(75,x_boston,y_boston)



    default_order = [
        ('MultiGaussClassify_WithFullMatrix', 'Boston50'),
        ('MultiGaussClassify_WithFullMatrix', 'Boston75'),
        ('MultiGaussClassify_WithFullMatrix', 'Digits'),
        ('MultiGaussClassify_WithDiagonal', 'Boston50'),
        ('MultiGaussClassify_WithDiagonal', 'Boston75'),
        ('MultiGaussClassify_WithDiagonal', 'Digits'),
        ('LogisticRegression', 'Boston50'),
        ('LogisticRegression', 'Boston75'),
        ('LogisticRegression', 'Digits')
    ]

    methods = {
        ('MultiGaussClassify_WithFullMatrix', 'Boston50'):
        (MultiGaussClassify(len(np.unique(y_boston_50)),x_boston_50.shape[1]), x_boston_50, y_boston_50),
        ('MultiGaussClassify_WithFullMatrix', 'Boston75'):
        (MultiGaussClassify(len(np.unique(y_boston_75)),x_boston_50.shape[1]), x_boston_75, y_boston_75),
        ('MultiGaussClassify_WithFullMatrix', 'Digits'):
        (MultiGaussClassify(len(np.unique(y_digits)),x_digits.shape[1]), x_digits, y_digits),
        ('MultiGaussClassify_WithDiagonal', 'Boston50'):
        (MultiGaussClassify(len(np.unique(y_boston_50)),x_boston_50.shape[1], True), x_boston_50, y_boston_50),
        ('MultiGaussClassify_WithDiagonal', 'Boston75'):
        (MultiGaussClassify(len(np.unique(y_boston_75)),x_boston_50.shape[1], True), x_boston_75, y_boston_75),
        ('MultiGaussClassify_WithDiagonal', 'Digits'):
        (MultiGaussClassify(len(np.unique(y_digits)),x_digits.shape[1], True), x_digits, y_digits),
        ('LogisticRegression', 'Boston50'):
        (LogisticRegression(), x_boston_50, y_boston_50),
        ('LogisticRegression', 'Boston75'):
        (LogisticRegression(), x_boston_75, y_boston_75),
        ('LogisticRegression', 'Digits'):
        (LogisticRegression(), x_digits, y_digits)
    }

    if dataset == 'all':
        order = default_order
    else:
        order = [(method_name, dataset)]

    for key in order:
        name, dataset = key
        method, X, y = methods[key]
        print('==============')
        print('method: {}, dataset: {}'.format(key[0], key[1]))
        scores = my_cross_val(method, X, y, k)
        report(name, dataset, scores,True)


if __name__ == '__main__':
    q3(sys.argv[1:])