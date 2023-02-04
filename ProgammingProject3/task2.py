import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from task1 import sigmoid,stopping_criteria,compute_SN,get_K,predictionI,sigmoid_ind


A_path = './pp3data/A.csv'
USPS_path = './pp3data/usps.csv'

AL_path = './pp3data/labels-A.csv'
USPSL_path = './pp3data/labels-usps.csv'

def BLR(train_x,train_y,test_x,test_y,W):
    alpha = 0.1
    X = test_x
    n = test_x.shape[0]
    S_N = compute_SN(train_x, W)
    y_pred = []
    err = []
    #print(test_y)
    y_orig = test_y[0]
    #print(y_orig[0],y_orig[1],y_orig[2])
    err_cnt = 0
    #print(test_x)
    for i,j in test_x.iterrows():
        #print(i,j)
        row = np.array(test_x.loc[i])
        mu = np.dot(W,row.T)
        var = np.dot(np.dot(row,S_N),row.T)
        k = get_K(var)
        #print(mu,var,k)
        y_pred = predictionI(sigmoid_ind(k*mu))

        if(y_pred == y_orig[i]):
            err.append(0)
        else:
            err.append(1)
            err_cnt+=1
    return err_cnt/test_x.shape[0]


def Newtons_Update(X,Y):
    w_n = np.array([0 for i in range(X.shape[1])])
    alpha = 0.1
    m = X.shape[1]
    n = X.shape[0]
    t = Y[0]

    iter = 0
    weights = []
    time_stamps = []
    prev_time = datetime.datetime.now()

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
            break
        weights.append(w_new)
        time_stamps.append((datetime.datetime.now() - prev_time).total_seconds())
        w_n = w_new

    return weights,time_stamps,w_new

def Gradient_Ascent(X,Y):
    w_n = np.array([0 for i in range(X.shape[1])])
    alpha = 0.1
    t = Y[0]
    theta = 0.001
    weights = []
    time_stamps = []
    prev_time = datetime.datetime.now()
    cnt = 0
    iter = 0

    while(1):
        iter+=1
        cnt+=1
        y = sigmoid(np.dot(w_n,X.T))

        alpha_w_n = np.array([alpha * i for i in w_n])
        K = np.dot(X.T, (y - t)) + alpha_w_n
        w_new = w_n - theta*K

        if(cnt == 10):
            weights.append(w_new)
            time_stamps.append((datetime.datetime.now() - prev_time).total_seconds())
            cnt = 0

        if (stopping_criteria(w_n, w_new) or iter >= 6000):
            break

        w_n = w_new

    return weights,time_stamps,w_new

def GradientAscent_LineSearch(X,Y,test_x,test_y):
    w_n = np.array([0 for i in range(X.shape[1])])
    alpha = 0.1
    t = Y[0]
    theta = 1
    gamma = 0.001
    weights = []
    time_stamps = []
    prev_time = datetime.datetime.now()
    cnt = 0
    iter = 0
    prev_loss = BLR(X,Y,test_x,test_y,w_n)

    while (1):
        iter += 1
        print(iter)
        cnt += 1
        y = sigmoid(np.dot(w_n, X.T))

        alpha_w_n = np.array([alpha * i for i in w_n])
        grad = np.dot(X.T, (y - t)) + alpha_w_n
        w_new = w_n - theta * grad

        curr_loss = BLR(X,Y,test_x,test_y,w_new)
        alpha_w_n = np.array([alpha * i for i in w_n])
        curr_grad = np.dot(X.T, (y - t)) + alpha_w_n

        while(curr_loss>prev_loss - gamma*theta*np.matmul(grad.T, grad)):
            theta = theta/2
            w_new = w_n - theta*grad
            curr_loss= BLR(X,Y,test_x,test_y,w_new)
            alpha_w_n = np.array([alpha * i for i in w_n])
            curr_grad = np.dot(X.T, (y - t)) + alpha_w_n

        if (stopping_criteria(w_n, w_new) or iter >=100):
            break

        if (cnt == 10):
            weights.append(w_new)
            time_stamps.append((datetime.datetime.now() - prev_time).total_seconds())
            cnt = 0
        prev_loss = curr_loss
        w_n = w_new

    return weights, time_stamps, w_new

def train_test_split(X,Y):
    N = X.shape[0]
    split = 2/3
    k = int(N*split)
    return X[:k], Y[:k], X[k:],Y[k:]

def plot_results(e_N,e_G,t_N,t_G,dataset):
    fig,ax = plt.subplots(1,2)
    ax[0].plot(t_N,e_N)
    ax[1].plot(t_G,e_G)
    ax[0].set_xlabel('Time in seconds (s)')
    ax[0].set_ylabel("Error")
    ax[1].set_xlabel('Time in seconds (s)')
    ax[1].set_ylabel("Error")
    ax[0].set_title('Newtons Update: '+str(dataset)+' dataset')
    ax[1].set_title('Gradient Ascent: '+str(dataset)+' dataset')
    plt.savefig('Task2_'+str(dataset)+'.png')
    return 0

def task2_helper(X,Y,dataset):
    X = pd.read_csv(X, header=None)
    Y = pd.read_csv(Y, header=None)

    N = X.shape[0]

    X.insert(loc=0, column='w_0', value=[1 for i in range(N)])
    train_x, train_y, test_x, test_y = train_test_split(X, Y)

    err = []
    T_N = []
    T_G = []

    for i in range(0,3):
        W_N, t_n, W = Newtons_Update(train_x,train_y)
        if i ==0:
            T_N=np.array(t_n)
        else:
            T_N = T_N + np.array(t_n)
        W_G, t_g, W = Gradient_Ascent(train_x,train_y)
        if i == 0:
            T_G=t_g
        else:
            T_G = T_G + np.array(t_g)

    T_N = [round((i/3),3) for i in T_N]
    T_G = [round((i/3),3) for i in T_G]

    err_N = []
    err_G = []

    print("Time taken for Newton's updates")
    print(T_N)
    print("Time taken for Gradient Ascent updates")
    print(T_G)

    for k in W_N:
        err_N.append(BLR(train_x, train_y, test_x, test_y, k))
    for k in W_G:
        err_G.append(BLR(train_x, train_y, test_x, test_y, k))
    print("Error for Newton's updates")
    print(err_N)
    print("Error for Gradient Ascent updates")
    print(err_G)

    plot_results(err_N,err_G,T_N,T_G,dataset)
    return 0


def Task_2():
    print("Computing for dataset-A")
    task2_helper(A_path,AL_path,"A")
    print("Plot is saved in the current directory with filename: Task2_A.png")
    print("Computing for dataset-USPS")
    task2_helper(USPS_path,USPSL_path,"USPS")
    print("Plot is saved in the current directory with filename: Task2_USPS.png")
    return

if __name__ == '__main__':
    Task_2()
