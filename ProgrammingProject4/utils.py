import numpy as np
import math
import matplotlib.pyplot as plt

import pandas as pd


def get_words(dataset,documents):
    D = list(range(1,documents+1))

    words = []
    for i in D:
        with open(dataset + str(i)) as doc:
            text = doc.readlines()[0]
            for word in text.split(' '):
                words.append(word)
    return words

def train_test_split_random(X,Y,split):
    rows = len(X)
    perm = np.random.permutation(rows)
    #print(perm)
    train_len = int(rows*split)
    #print(train_len,len(X))
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    cnt = 0
    #print(len(X),len(Y))
    for ind in perm:
        train_X.append(X[ind])
        train_Y.append(Y[ind])
        cnt+=1
        if(cnt>train_len):
            break
    for i in range(cnt,rows):
        test_X.append(X[perm[i]])
        test_Y.append(Y[perm[i]])
    #print(len(train_X),len(train_Y),len(test_X),len(test_Y))
    return train_X,train_Y,test_X,test_Y

def sample_trainR(train_x,train_y,split):
    rows = int(len(train_x)*split)
    perm = np.random.permutation(rows)
    new_train_x = []
    new_train_y = []
    for i in perm:
        new_train_x.append(train_x[perm[i]])
        new_train_y.append(train_y[perm[i]])
    return new_train_x,new_train_y

def sigmoid_ind(a):
    try:
        pred = (1/(1+math.exp(-a)))
    except OverflowError:
        pred = (float('inf'))
    return pred

def get_K(a):
    K = np.sqrt(1+(3.14*a/8))
    return 1/K

def compute_SN(X,W):
    N = X.shape[0]
    m = X.shape[1]
    alpha = 0.1

    l = np.dot(W, X.T)
    y = np.array(sigmoid(l))
    R = np.diag(y * (1 - y))

    S_N = np.linalg.inv(alpha*np.identity(m) + np.dot(np.dot(X.T,R),X))

    return S_N

def predictionI(a):
    if a>=0.5:
        return 1
    else:
        return 0

def sigmoid(a):
    pred = []
    for i in a:
        try:
            pred.append(1 / (1 + math.exp(-i)))
        except OverflowError:
            pred.append(float('inf'))
    return pred

def stopping_criteria(w_n,w_new):
    if np.linalg.norm(w_n) == 0:
        return 0
    K = (np.linalg.norm(w_new - w_n) / np.linalg.norm(w_n))
    if(K<0.001):
        return 1
    return 0

def plotresults(x,y1,y2,y3,y4):
    plt.plot(x,y1)
    plt.errorbar(x,y1,yerr = y2,label = 'LDA',color = 'red')
    plt.plot(x,y3)
    plt.errorbar(x,y3,yerr = y4,label = 'BOW',color = 'blue')
    plt.legend()
    plt.xlabel('Training Size')
    plt.ylabel('Error')
    plt.title('Comparison of LDA & BOW')
    plt.savefig('Task2.png')
    plt.close()

def Newtons_Update(X,Y,alpha):
    X = pd.DataFrame(X)
    #Y = pd.DataFrame(Y)
    #print(Y)
    w_n = np.array([0 for i in range(X.shape[1])])
    #print(X)
    m = X.shape[1]
    #n = X.shape[0]
    #print(Y)
    t = Y

    #print(X.shape,len(Y))
    iter = 0
    #prev_time = datetime.datetime.now()

    while(1):
        iter += 1
        ## compute y
        l = np.dot(w_n,X.T)
        y = np.array(sigmoid(l))
        R = np.diag(y*(1-y))

        alpha_w_n = np.array([alpha*i for i in w_n])

        K = np.dot(X.T,(y-t)) + alpha_w_n
        S_N = np.linalg.inv(alpha*np.identity(m) + np.dot(np.dot(X.T,R),X))
        w_new = w_n - np.dot(np.linalg.inv(alpha*np.identity(m) + np.dot(np.dot(X.T,R),X)),K)

        if(stopping_criteria(w_n,w_new) or iter>=100):
            #print(w_n)
            break
        #time_stamps.append((datetime.datetime.now() - prev_time).total_seconds())
        #print("Iteration",iter)
        w_n = w_new

    return w_n

def BLR(train_x,train_y,test_x,test_y,W):
    train_x = pd.DataFrame(train_x)
    #train_y = pd.DataFrame(train_y)
    test_x = pd.DataFrame(test_x)
    #test_y = pd.DataFrame(test_y)
    alpha = 0.1
    X = test_x
    n = test_x.shape[0]
    S_N = compute_SN(train_x, W)
    y_pred = []
    err = []
    y_orig = test_y
    err_cnt = 0

    for i,j in test_x.iterrows():
        #print(i,j)
        row = np.array(test_x.loc[i])
        #print(row.shape,"r")
        #print(W)
        #print(row)
        mu = np.dot(W,row.T)
        var = np.dot(np.dot(row,S_N),row.T)
        k = get_K(var)
        #print(k,mu)
        y_pred = predictionI(sigmoid_ind(k*mu))
        #print(y_pred,"Pred")
        #print(y_orig[i],"Orig")
        if(y_pred == y_orig[i]):
            err.append(0)
        else:
            err.append(1)
            err_cnt+=1
    #print(err_cnt, "Error Cnt")
    return err_cnt/test_x.shape[0]