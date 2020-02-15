import numpy as np
import random



def mix_data(features,classes):
    if len(classes.shape) == 1:
        classes = classes.reshape(len(classes), 1)
    data = np.append(features,classes,axis=1)
    np.random.shuffle(data)
    return data

def partioning_data(data,k):    
    new_data = np.array_split(data,k)
    return new_data

def split_validation(folds,i):
    dataset = folds[:]
    validation = dataset.pop(i)
    return np.concatenate(dataset), validation

def error_rate (predict,target):
    predict = predict.reshape(len(predict),1)
    target = target.reshape(len(target),1)
    
    return np.sum(predict==target)/float(predict.shape[0])

def train_model(method,train,validation):
    x_train = train[:,:-1]
    y_train = train[:,-1:]
    x_validation = validation[:,:-1]
    y_validation = validation[:,-1:]
    method.fit(x_train,y_train.ravel())
    predict = method.predict(x_validation)
    
    return error_rate(predict,y_validation)


def my_cross_val(method,X,y,k):
    score_list = []
    dataset = mix_data(X,y)
    new_dataset = partioning_data(dataset,k)
    for fold in range(len(new_dataset)):
        train,validation = split_validation(new_dataset,fold)
        score = train_model(method,train,validation)
        score_list.append(score)
        
    # mean_score = np.mean(score_list)
    # std_score = np.std(score_list)
    # score = [{'score_list':[score_list]}, {'mean_score': mean_score}, {'std_score':std_score}]
    
    return score_list
