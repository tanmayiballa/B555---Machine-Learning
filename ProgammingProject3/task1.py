import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

A_path = './pp3data/A.csv'
B_path = './pp3data/B.csv'
iris_x = './pp3data/irlstest.csv'

AL_path = './pp3data/labels-A.csv'
BL_path = './pp3data/labels-B.csv'
iris_y = './pp3data/labels-irlstest.csv'

USPS_path = './pp3data/usps.csv'
USPSL_path = './pp3data/labels-usps.csv'

def train_test_split(X,Y):
    N = X.shape[0]
    col = X.shape[1]
    X[col] = Y[0]
    test = X.sample(frac = 0.34)
    condition = X.isin(test)
    index = []
    for i,j in condition.iterrows():
        flag = 1
        for k in j:
            if k == False:
                flag = 0
        if flag == 1:
            index.append(i)
    X.drop(index = index,inplace = True)
    train_y = X[col]
    train_x = X.drop(columns=[col])
    test_y = test[col]
    test_x = test.drop(columns = [col])

    return train_x.reset_index().drop(columns=['index']),train_y.reset_index().drop(columns=['index']),test_x.reset_index().drop(columns=['index']),test_y.reset_index().drop(columns=['index'])

def train_split(X,Y,size):
    N = X.shape[0]
    return X[:int(size*N)],Y[:int(size*N)]

def sigmoid_ind(a):
    try:
        pred = (1/(1+math.exp(-a)))
    except OverflowError:
        pred = (float('inf'))
    return pred

def sigmoid(a):
    pred = []
    for i in a:
        try:
            pred.append(1 / (1 + math.exp(-i)))
        except OverflowError:
            pred.append(float('inf'))
    return pred

def mse(y_pred,y_orig):
    err = 0
    for i in range(len(y_pred)):
        err+=math.pow(y_pred[i] - y_orig[i],2)
    return (err/len(y_orig))

def find_mse(pred_y, test_y):
    N = len(pred_y)
    err = 0.0
    diff = np.array(pred_y) - np.array(test_y)
    square_diff = diff*diff
    err = np.sum(square_diff)
    return (err/N)

def find_err(pred_y,test_y):
    cnt = 0
    N = len(pred_y)
    for i in pred_y:
        if(pred_y[i]!=test_y[i]):
            cnt+=1
    return cnt/N

def predictionI(a):
    if a>=0.5:
        return 1
    else:
        return 0

def prediction(a):
    pred = []
    for i in a:
        if(i>=0.5):
            pred.append(1)
        else:
            pred.append(0)
    return pred

def plotresults(x,y1,y2,y3,y4,dataset):
    plt.plot(x,y1)
    plt.errorbar(x,y1,yerr = y2,label = 'Mean&stdGenerative')
    plt.plot(x,y3)
    plt.errorbar(x,y3,yerr = y4,label = 'Mean&stdBayesianLogReg')
    plt.legend()
    plt.xlabel('Training Size')
    plt.ylabel('Error')
    plt.title('Comparison of Generative & Bayesian LogReg for '+dataset+' dataset')
    plt.savefig('Task1_'+str(dataset)+'.png')
    plt.close()

def Generative_Model(train_x,train_y,test_x,test_y):
    N = train_x.shape[0]
    N1 = 0
    N2 = 0
    for index, j in train_y.iterrows():
        if(list(j)[0] == 1):
            N1+=1
        else:
            N2+=1
    P1 = N1/N
    P2 = N2/N

    mu_1 = [0 for i in range(0,train_x.shape[1])]
    mu_2 = [0 for i in range(0,train_x.shape[1])]
    col = train_y.columns

    for index, j in train_y.iterrows():
        t = train_y.iloc[index][col[0]]
        k = train_x.iloc[index]
        if t == 1:
            mu_1 = mu_1 + k
        else:
            mu_2 = mu_2 + k

    mu_1 = mu_1/N1
    mu_2 = mu_2/N2

    ## Find S1 and S2

    S1 = 0
    S2 = 0

    for index,j in train_x.iterrows():
        t = train_y.iloc[index][col[0]]
        k = train_x.iloc[index]
        if t == 1:
            k1 = k - mu_1
            S1 += np.dot(k1,k1.T)
        else:
            k0 = k - mu_2
            S2 += np.dot(k0,k0.T)
    S1 = S1/N1
    S2 = S2/N2
    S = (N1/N)*S1 + (N2/N)*S2
    sigma_inv = S

    W = sigma_inv*(mu_1 - mu_2)

    W0 = -0.5*np.dot(mu_1.T,(sigma_inv*mu_1)) + 0.5*(np.dot(mu_2.T,(sigma_inv*mu_2))) + math.log(P1/P2)

    y_pred = prediction(sigmoid(np.dot(W.T,test_x.T) + W0))
    t_y = np.array(test_y[col[0]])

    err = find_mse(y_pred,t_y)
    return err

def stopping_criteria(w_n,w_new):
    if np.linalg.norm(w_n) == 0:
        return 0
    K = (np.linalg.norm(w_new - w_n) / np.linalg.norm(w_n))
    if(K<0.001):
        return 1
    return 0

def compute_weights(X,Y):
    w_n = np.array([0 for i in range(X.shape[1])])
    alpha = 0.1
    m = X.shape[1]
    n = X.shape[0]
    col = Y.columns
    t = Y[col[0]]

    iter = 0

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
        w_n = w_new

    return w_new

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

def BayesianLogisticR(train_x,train_y,test_x,test_y):
    alpha = 0.1
    X = test_x
    n = test_x.shape[0]

    W = compute_weights(train_x, train_y)

    S_N = compute_SN(train_x, W)

    y_pred = []
    err = []
    col = train_y.columns
    y_orig = test_y[col[0]]
    err_cnt = 0

    for i,j in test_x.iterrows():
        row = np.array(test_x.iloc[i])
        mu = np.dot(W,row.T)
        var = np.dot(np.dot(row,S_N),row.T)
        k = get_K(var)
        y_pred = predictionI(sigmoid_ind(k*mu))

        if(y_pred == y_orig[i]):
            err.append(0)
        else:
            err.append(1)
            err_cnt+=1

    return err_cnt/test_x.shape[0]

def Task_1A(X,Y):
    X = pd.read_csv(X, header=None)
    Y = pd.read_csv(Y, header=None)
    train_x,train_y,test_x,test_y = train_test_split(X,Y)
    err = []

    training_size = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for i in range(0,10):
        tr_x, tr_y = train_split(train_x,train_y,training_size[i])
        err.append(Generative_Model(tr_x,tr_y,test_x,test_y))
    return err

def Task_1B(X,Y):

    X = pd.read_csv(X, header=None)
    Y = pd.read_csv(Y, header=None)

    N = X.shape[0]
    # Inserting w_0 feature with all 1's for the dataset

    X.insert(loc=0, column = 'w_0', value=[1 for i in range(N)])

    train_x,train_y,test_x,test_y = train_test_split(X,Y)

    err = []

    training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(0, 10):
        tr_x, tr_y = train_split(train_x, train_y, training_size[i])
        err.append(BayesianLogisticR(tr_x, tr_y, test_x, test_y))
    return err

def Task_1_helper(X,Y,dataset):
    iter = 0
    err_mapG = {}
    err_mapB = {}
    training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(0,10):
        err_mapG[i] = list()
        err_mapB[i] = list()

    ## Loop for 30 iterations

    while(1):
        iter+=1
        print("Iteration: ",iter)
        err_G = Task_1A(X,Y) # Generative_Model
        err_B = Task_1B(X,Y) # Bayesian Logistic Regression

        for i in range(0,10):
            err_mapG[i].append(err_G[i])
            err_mapB[i].append(err_B[i])

        if(iter>=30):
            break

    meanG = []
    meanB = []
    stdB = []
    stdG = []

    for i in range(0,10):
        meanG.append(np.mean((err_mapG[i])))
        meanB.append(np.mean((err_mapB[i])))

        stdG.append(np.std(err_mapG[i]))
        stdB.append(np.std(err_mapB[i]))

    plotresults(training_size,meanG,stdG,meanB,stdB,dataset)

    return meanG,stdG,meanB,stdB

def Task1():
    res_df = {}
    training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("Computing for Dataset - A")

    meanG,stdG,meanB,stdB = Task_1_helper(A_path,AL_path,"A")

    res_df['training_size'] = training_size
    res_df['A_MeanError_Generative'] = meanG
    res_df['A_std_Generative'] = stdG
    res_df['A_MeanError_Bayesian'] = meanB
    res_df['A_std_Bayesian'] = stdB

    print("Plot saved in the current directory with name: Task_1_A.png")

    print("Computing for Dataset - B")

    meanG,stdG,meanB,stdB = Task_1_helper(B_path, BL_path, "B")

    res_df['B_MeanError_Generative'] = meanG
    res_df['B_std_Generative'] = stdG
    res_df['B_MeanError_Bayesian'] = meanB
    res_df['B_std_Bayesian'] = stdB

    print("Plot saved in the current directory with name: Task_1_B.png")

    print("Computing for Dataset - USPS")

    meanG,stdG,meanB,stdB = Task_1_helper(USPS_path, USPSL_path, "USPS")


    res_df['USPS_MeanError_Generative'] = meanG
    res_df['USPS_std_Generative'] = stdG
    res_df['USPS_MeanError_Bayesian'] = meanB
    res_df['USPS_std_Bayesian'] = stdB

    print("Plot saved in the current directory with name: Task_1_USPS.png")

    result_df = pd.DataFrame(res_df)
    print(tabulate(result_df, headers=result_df.columns, tablefmt='grid'))
    return

if __name__ == '__main__':
    Task1()


