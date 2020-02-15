import numpy as np
import random


def split_with_pi(dataset,percentage):
    new_dataset = np.random.shuffle(dataset)
    train_index = int(np.floor(dataset.shape[0]*percentage))
    features = dataset[:,:-1]
    classes = dataset[:,-1:]
    x_train = features[:train_index]
    y_train = classes[:train_index]
    x_validation = features[train_index:]
    y_validation = classes[train_index:]
    
    return x_train,y_train,x_validation,y_validation

def accuracy_rate (predict,target):
    predict = predict.reshape(len(predict),1)
    target = target.reshape(len(target),1)
    
    return np.sum(predict==target)/float(predict.shape[0])

def mix_data(features,classes):
    if len(classes.shape) == 1:
        classes = classes.reshape(len(classes), 1)
    data = np.append(features,classes,axis=1)
    np.random.shuffle(data)
    return data

def train_model_with_pi(method,x_train,y_train,x_validation,y_validation):
    method.fit(x_train,y_train.ravel())
    predict = method.predict(x_validation)
    
    return accuracy_rate(predict,y_validation)


def my_train_test(method,X,y,pi,k):
    score_list = []
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1) 
    dataset = mix_data(X,y)
    for i in range(k):
        x_train,y_train,x_validation,y_validation = split_with_pi(dataset,pi)
        score = train_model_with_pi(method,x_train,y_train,x_validation,y_validation)
        score_list.append(score)

    # mean_score = np.mean(score_list)
    # std_score = np.std(score_list)
    
    # score = [{'score_list':[score_list]}, {'mean_score': mean_score}, {'std_score':std_score}]
    
    return score_list