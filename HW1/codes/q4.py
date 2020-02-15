from sklearn.datasets import load_digits
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC



from datasets import load_dataset


from my_cross_val import my_cross_val
from proj import (
    rand_proj,
    quad_proj
    )

from utils import (
    report,
    wrapper_args
    )

import sys


def q4(argv):

    x_digits, y_digits = load_dataset(load_digits)

    dataset, method_name, k, latex = wrapper_args(
        argv, 'q4', ['X1', 'X2', 'X3'])

    X1 = rand_proj(x_digits, 32)
    X2 = quad_proj(x_digits)
   
    default_order = [
        ('LinearSVC', 'X1'),
        ('LinearSVC', 'X2'),
        ('SVC', 'X1'),
        ('SVC', 'X2'),
        ('LogisticRegression', 'X1'),
        ('LogisticRegression', 'X2'),
        ]

    methods = {('LinearSVC', 'X1'):
               (LinearSVC(max_iter=2000), X1, y_digits),
               ('LinearSVC', 'X2'):
               (LinearSVC(max_iter=2000), X2, y_digits),
               ('SVC', 'X1'):
               (SVC(gamma='scale', C=10), X1, y_digits),
               ('SVC', 'X2'):
               (SVC(gamma='scale', C=10), X2, y_digits),
               ('LogisticRegression', 'X1'):
               (LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000), X1, y_digits),
               ('LogisticRegression', 'X2'):
               (LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000), X2, y_digits)}

    if dataset == 'all':
        order = default_order
    else:
        order = [(method_name, dataset)]

    for key in order:
        name, dataset = key
        method, X, y = methods[key]
        print('==============')
        print('method: {}, dataset: {}'.format(name, dataset))
        scores = my_cross_val(method, X, y, k)
        report(name, dataset, scores, latex=latex)


if __name__ == '__main__':
    q4(sys.argv[1:])