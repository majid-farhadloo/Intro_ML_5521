from sklearn.datasets import load_boston, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


from datasets import (
    percentileAssignment,
    load_dataset
    )

from my_train_test import my_train_test
from utils import (
    report,
    wrapper_args
    )

import sys


def q3ii(argv=None):
  dataset, method_name, k, pi, latex = wrapper_args(
  argv, 'q3ii', ['Boston50', 'Boston75', 'Digits'], include_pi=True)

  x_boston, y_boston = load_dataset(load_boston)
  x_digits, y_digits = load_dataset(load_digits)
  x_boston_50, y_boston_50 = percentileAssignment(50,x_boston,y_boston)
  x_boston_75, y_boston_75 = percentileAssignment(75,x_boston,y_boston)

  default_order = [
  ('LinearSVC', 'Boston50'),
  ('LinearSVC', 'Boston75'),
  ('LinearSVC', 'Digits'),
  ('SVC', 'Boston50'),
  ('SVC', 'Boston75'),
  ('SVC', 'Digits'),
  ('LogisticRegression', 'Boston50'), 
  ('LogisticRegression', 'Boston75'),
  ('LogisticRegression', 'Digits')]


  methods = {('LinearSVC', 'Boston50'):
  (LinearSVC(max_iter=2000), x_boston_50, y_boston_50),
  ('LinearSVC', 'Boston75'):
  (LinearSVC(max_iter=2000), x_boston_75, y_boston_75),
  ('LinearSVC', 'Digits'):
  (LinearSVC(max_iter=2000), x_digits, y_digits),
  ('SVC', 'Boston50'):
  (SVC(gamma='scale', C=10), x_boston_50, y_boston_50),
  ('SVC', 'Boston75'):
  (SVC(gamma='scale', C=10), x_boston_75, y_boston_75),
  ('SVC', 'Digits'):
  (SVC(gamma='scale', C=10), x_digits, y_digits),
  ('LogisticRegression', 'Boston50'):
  (LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000), x_boston_50, y_boston_50),
  ('LogisticRegression', 'Boston75'):
  (LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000), x_boston_75, y_boston_75),
  ('LogisticRegression', 'Digits'):
  (LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000), x_digits, y_digits)}


  if dataset == 'all':
    order = default_order
  else:
    order = [(method_name, dataset)]

  for key in order:
    name, dataset = key
    method, X, y = methods[key]
    print('==============')
    print('method: {}, dataset: {}'.format(key[0], key[1]))
    scores = my_train_test(method, X, y, 0.75, k)
    report(name, dataset, scores, latex=latex)


if __name__ == '__main__':
    q3ii(sys.argv[1:])