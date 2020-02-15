Author: Majid Farhadloo
CSCI 5521, intro to ML.
Email: farha043@umn.edu

**************************************************************
This file explains the rquired functions in this homework, and you can run the program.

**************************************************************

Python version: $ python3 -V
Python 3.8.1

**************************************************************


Running the assignment as the follows:

There are 2 questions for two type of fucntions. Please refer to the homework questions regarding the functionalities.
it can be find in CSci_5521_HW1.pdf

**************************************************************

Running the q3i.py:

For q3i, simply run python q3i.py. By default, all combinations asked for in the assignment will be run.
Three fiting models are as follows:
LogisticRegression
LinearSVC
SVC

if you want to run datasets individually, it can be done with following commands:

$ python q3i.py -h
python q3i.py -d <Boston50|Boston75|Digits> -m <LinearSVC|SVC|LogisticRegression> -k [num folds] -l [latex output on]

same for the q3ii.py

**************************************************************

For q3ii, simply run python q3ii.py. By default, all combinations asked for in the assignment will be run.
Outout will print it out into the console.

fitting models are imported from the sklearn library and it is same as the q3i.py 

**************************************************************

For q4, simply run python q4.py. By default, all combinations asked for in the assignment will be run
Outout will print it out into the console.


fitting models are imported from the sklearn library and it is same as the q3i.py

if you want to run datasets individually, it can be done with following commands:
python q4.py -h
python q4.py -d <X1|X2|X3> -m <LinearSVC|SVC|LogisticRegression> -k [num folds] -l [latex output on]

**************************************************************


Sample running for q3i.py:

method: LinearSVC, dataset: Boston50

```

name: LinearSVC
dataset: Boston50
error rates:
0.2157
0.1373
0.0784
0.2745
0.2353
0.2157
0.1600
0.3800
0.1200
0.2200
mean: 0.2037
std dev: 0.0817

**************************************************************

method: LinearSVC, dataset: Boston75
name: LinearSVC
dataset: Boston75
error rates:
0.0588
0.1373
0.2157
0.6667
0.4902
0.1569
0.2800
0.1200
0.0800
0.6000
mean: 0.2805
std dev: 0.2122

**************************************************************

method: LinearSVC, dataset: Digits
name: LinearSVC
dataset: Digits
error rates:
0.0333
0.0667
0.0667
0.0389
0.0444
0.0833
0.0333
0.0391
0.0615
0.0335
mean: 0.0501
std dev: 0.0170

**************************************************************


method: SVC, dataset: Boston50
name: SVC
dataset: Boston50
error rates:
0.1569
0.2549
0.2353
0.1961
0.2353
0.2941
0.2000
0.2600
0.3200
0.1600
mean: 0.2313
std dev: 0.0511

**************************************************************


method: SVC, dataset: Boston75
name: SVC
dataset: Boston75
error rates:
0.2549
0.2745
0.1569
0.1961
0.2745
0.2941
0.1200
0.3200
0.2000
0.2600
mean: 0.2351
std dev: 0.0608

**************************************************************


method: SVC, dataset: Digits
name: SVC
dataset: Digits
error rates:
0.0111
0.0056
0.0111
0.0000
0.0000
0.0111
0.0111
0.0056
0.0168
0.0168
mean: 0.0089
std dev: 0.0057

**************************************************************


method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.1176
0.1373
0.2157
0.1373
0.1569
0.1373
0.1000
0.1000
0.2000
0.1200
mean: 0.1422
std dev: 0.0370

**************************************************************


method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.1176
0.0588
0.0588
0.0784
0.0980
0.1176
0.1400
0.1000
0.1000
0.1400
mean: 0.1009
std dev: 0.0277

**************************************************************


method: LogisticRegression, dataset: Digits
name: LogisticRegression
dataset: Digits
error rates:
0.0222
0.0278
0.0444
0.0333
0.0389
0.0222
0.0389
0.0335
0.0223
0.0335
mean: 0.0317
std dev: 0.0075

```

Sample running for the question 3ii:


```
python3 q3ii.py
==============
method: LinearSVC, dataset: Boston50
name: LinearSVC
dataset: Boston50
error rates:
0.3543
0.4646
0.3228
0.1654
0.1102
0.2283
0.2283
0.4567
0.1732
0.1890
mean: 0.2693
std dev: 0.1177
==============
method: LinearSVC, dataset: Boston75
name: LinearSVC
dataset: Boston75
error rates:
0.1260
0.0945
0.4173
0.1102
0.0945
0.2520
0.0787
0.2520
0.6142
0.1024
mean: 0.2142
std dev: 0.1680
==============
method: LinearSVC, dataset: Digits
name: LinearSVC
dataset: Digits
error rates:
0.0533
0.0600
0.0489
0.0489
0.0489
0.0778
0.0333
0.0333
0.0800
0.0556
mean: 0.0540
std dev: 0.0149
==============
method: SVC, dataset: Boston50
name: SVC
dataset: Boston50
error rates:
0.2835
0.2205
0.2520
0.2441
0.3228
0.2205
0.2362
0.2205
0.2598
0.2283
mean: 0.2488
std dev: 0.0313
==============
method: SVC, dataset: Boston75
name: SVC
dataset: Boston75
error rates:
0.2520
0.2441
0.1969
0.2520
0.1575
0.1654
0.2126
0.2126
0.3307
0.1890
mean: 0.2213
std dev: 0.0483
==============
method: SVC, dataset: Digits
name: SVC
dataset: Digits
error rates:
0.0133
0.0089
0.0111
0.0067
0.0111
0.0133
0.0022
0.0156
0.0156
0.0133
mean: 0.0111
std dev: 0.0040
==============
method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.1811
0.1496
0.1339
0.1024
0.1260
0.0866
0.1024
0.0787
0.0945
0.1339
mean: 0.1189
std dev: 0.0302
==============
method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.0945
0.0709
0.1024
0.1024
0.1339
0.0866
0.0866
0.1339
0.1417
0.0945
mean: 0.1047
std dev: 0.0226
==============
method: LogisticRegression, dataset: Digits
name: LogisticRegression
dataset: Digits
error rates:
0.0356
0.0356
0.0422
0.0311
0.0267
0.0311
0.0400
0.0311
0.0244
0.0244
mean: 0.0322
std dev: 0.0058

```
Sample running for the question 4:

```
==============
method: LinearSVC, dataset: X1
name: LinearSVC
dataset: X1
error rates:
0.1000
0.0833
0.0778
0.1000
0.0611
0.0611
0.0889
0.1117
0.0782
0.1061
mean: 0.0868
std dev: 0.0168
==============
method: LinearSVC, dataset: X2
name: LinearSVC
dataset: X2
error rates:
0.0167
0.0111
0.0111
0.0167
0.0111
0.0056
0.0111
0.0056
0.0056
0.0335
mean: 0.0128
std dev: 0.0079
==============
method: SVC, dataset: X1
name: SVC
dataset: X1
error rates:
0.0111
0.0222
0.0222
0.0111
0.0278
0.0278
0.0167
0.0168
0.0056
0.0112
mean: 0.0172
std dev: 0.0072
==============
method: SVC, dataset: X2
name: SVC
dataset: X2
error rates:
0.0056
0.0111
0.0167
0.0111
0.0167
0.0222
0.0000
0.0168
0.0000
0.0112
mean: 0.0111
std dev: 0.0070
==============
method: LogisticRegression, dataset: X1
name: LogisticRegression
dataset: X1
error rates:
0.0556
0.0944
0.0778
0.0944
0.0889
0.0556
0.0778
0.0894
0.0782
0.0279
mean: 0.0740
std dev: 0.0204
==============
method: LogisticRegression, dataset: X2
name: LogisticRegression
dataset: X2
error rates:
0.0111
0.0056
0.0167
0.0278
0.0000
0.0167
0.0111
0.0056
0.0056
0.0168
mean: 0.0117
std dev: 0.0076

```

