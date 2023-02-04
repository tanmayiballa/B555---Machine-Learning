## Task-1

import os
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import random
import math

data_folder = os.path.join(os.getcwd(), '../pp2data')

lambda_val = 0

train_crime = '/train-crime.csv'
trainR_crime = '/trainR-crime.csv'
test_crime = '/test-crime.csv'
testR_crime = '/testR-crime.csv'

train_housing = '/train-housing.csv'
trainR_housing = '/trainR-housing.csv'
test_housing = '/test-housing.csv'
testR_housing = '/testR-housing.csv'

## Datasets for Task-2

train_f3 = '/train-f3.csv'
train_f5 = '/train-f5.csv'

trainR_f3 = '/trainR-f3.csv'
trainR_f5 = '/trainR-f3.csv'

test_f3 = '/test-f3.csv'
test_f5 = '/test-f5.csv'

testR_f3 = '/testR-f3.csv'
testR_f5 = '/testR-f5.csv'


def train_split(train_x,train_y,split_value):
    len = train_x.shape[0]
    train_split = int(split_value*len)
    print(train_split)
    return train_x[:train_split], train_y[:train_split]

def load_dataset(trainp_x,trainp_y,testp_x,testp_y):

    ## Loading dataset

    train_x = pd.read_csv(data_folder + trainp_x, header = None)
    train_y = pd.read_csv(data_folder + trainp_y, header = None)

    test_x = pd.read_csv(data_folder + testp_x, header = None)
    test_y = pd.read_csv(data_folder + testp_y, header = None)

    return train_x, train_y, test_x, test_y

def calc_piTpi(data):
    new_data = np.dot(np.transpose(data), data)
    return new_data

def calc_weights(data, target, lambda_val):
    TCA = calc_piTpi(data)

    identity = np.identity(TCA.shape[0])
    lambda_matrix = lambda_val*identity

    #print("Lambda_Matrix")
    #print(TCA)

    #tca = np.dot(np.linalg.inv(np.dot(np.transpose(data),data)),(np.dot(np.transpose(data),target)))

    A_inv = lambda_matrix + TCA

    A = np.linalg.inv(A_inv)

    B = np.dot(np.transpose(data),target)
    #print(B.shape)

    weights = np.dot(A,B)
    #print(weights.shape)
    return weights

def find_mse(pred_y, test_y):
    N = len(pred_y)
    err = 0.0

    diff = pred_y - test_y
    square_diff = diff*diff
    err = np.sum(square_diff)
    return (err/N)

def predict_y(data, w):

    res = np.dot(data, w)
    #print(res.shape)
    return res

def MLE(train_x, train_y,test_x,test_y, lambda_val):
    w = calc_weights(train_x, train_y, lambda_val)
    print(w, "Weights ", lambda_val)
    y_pred = predict_y(test_x, w)
    err = find_mse(y_pred, test_y)
    return err

def Model_Selection_new(train_x,train_y):
    alpha = random.randint(1,10)
    beta = random.randint(1,10)

    new_alpha = alpha
    new_beta = beta

    BpiT = beta*(np.transpose(train_x))
    BpiTpi = np.matmul(BpiT,train_x)
    eigen_value, eigen_vec = np.linalg.eig(BpiTpi)
    N = train_x.shape[0]
    it = 0

    while(True):
        print("Iteration: ", it)
        # print(alpha, beta)
        TCA = calc_piTpi(train_x)

        identity = np.identity(TCA.shape[0])
        lambda_matrix = alpha * identity

        S_N_inv = lambda_matrix + BpiTpi
        S_N = np.linalg.inv(S_N_inv)

        Bs_n = beta*S_N
        Bs_n_piT = np.matmul(Bs_n,np.transpose(train_x))

        m_N = np.dot(Bs_n_piT,train_y)

        gamma_vec = eigen_value / (eigen_value + alpha)
        gamma = np.round(np.sum(gamma_vec), 2)

        new_alpha = (gamma/np.matmul(m_N.T,m_N))

        #sumB = np.sum((train_y - np.matmul(m_N.T,train_x))**2)
        #new_beta = (N - gamma)/sumB

        phMn = (np.dot(train_x, m_N) - train_y)
        print(m_N.shape, train_x.shape, train_y.shape)

        new_beta = (N - gamma) / np.square(np.linalg.norm(phMn))

        print(new_alpha,new_beta)

        if ((new_beta - beta) <= 0.1 and (new_alpha - alpha) <= 0.1):
            # print(new_alpha, new_beta)
            print("End of iteration")
            break
        alpha = np.round(new_alpha, 3)
        beta = np.round(new_beta, 3)
        it += 1

    return new_alpha, new_beta

def Model_Selection(train_x, train_y):
    alpha = random.randint(1,10)
    beta = random.randint(1,10)

    new_alpha = alpha
    new_beta = beta

    N = train_x.shape[0]
    it = 0

    while(True):
        print("Iteration: ", it)
        #print(alpha, beta)
        TCA = calc_piTpi(train_x)

        Lambda = beta * (np.dot(np.transpose(train_x), train_x))
        eigen_value, eigen_vec = np.linalg.eig(Lambda)

        identity = np.identity(TCA.shape[0])
        lambda_matrix = alpha * identity

        S_N_inv =  lambda_matrix + beta*TCA
        S_N = np.linalg.inv(S_N_inv)

        ## Calculating m_N

        #piTt = np.dot(np.transpose(train_x),train_y)

        m_N = np.dot(np.dot((beta*S_N), np.transpose(train_x)), train_y)

        #m_N = beta*(np.dot(S_N,piTt))


        ## Calculating gamma:

        gamma_vec = eigen_value/(eigen_value+alpha)
        gamma = np.round(np.sum(gamma_vec),2)
        #print(gamma)

        new_alpha = (gamma / np.matmul(m_N.T, m_N))
        new_alpha = new_alpha[0][0]

        #new_alpha = np.round(gamma/np.square((np.linalg.norm(m_N))),2)

        #print(np.dot(np.transpose(m_N),m_N), "Hiii")
        #print(np.square((np.linalg.norm(m_N))), "Hiiiiiii1231")

        #new_alpha = gamma/np.dot(np.transpose(m_N),m_N)

        # sumB = np.sum((train_y - np.matmul(m_N.T,train_x))**2)
        # new_beta = (N - gamma)/sumB

        phMn = (np.dot(train_x, m_N) - train_y)
        #print(m_N.shape, train_x.shape, train_y.shape)

        new_beta = (N - gamma) / np.square(np.linalg.norm(phMn))

        #new_beta = (N - gamma)/np.square(np.linalg.norm(phMn))

        #print(beta_N, beta, "Compare")

        if((new_beta - beta) <=0.1 and (new_alpha - alpha)<=0.1):
            #print(new_alpha, new_beta)
            print("End of iteration")
            break
        alpha = np.round(new_alpha,4)
        beta = np.round(new_beta,4)
        it+=1

    return new_alpha, new_beta

def Bayesian_Regression(train_x, train_y,test_x, test_y):
    alpha, beta = Model_Selection(train_x,train_y)
    lambda_val = alpha/beta
    mse = MLE(train_x,train_y,test_x,test_y,lambda_val)
    return mse

def compute_logevidence(M,N,alpha,beta,train_x,train_y):
    term1 = float((M/2)*math.log(alpha))

    term2 = float((N/2)*math.log(beta))

    TCA = calc_piTpi(train_x)

    identity = np.identity(TCA.shape[0])
    lambda_matrix = alpha * identity

    A = lambda_matrix + beta * TCA
    A_inv = np.linalg.inv(A)

    m_N = np.dot(np.dot((beta * A_inv), np.transpose(train_x)), train_y)

    term3 = float((beta/2)*np.square((np.linalg.norm(train_y - (np.dot(train_x,m_N))))) + (alpha/2)*calc_piTpi(m_N))

    term4 = float(0.5*math.log(np.linalg.det(A)))


    term5 = float((N/2)*math.log(2*3.14))

    L_evid = term1 + term2 - term3 - term4 - term5

    return L_evid

def task_1_helper(mle_lambda):
    crime_dataset = load_dataset(train_crime, trainR_crime, test_crime, testR_crime)
    housing_dataset = load_dataset(train_housing,trainR_housing,test_housing,testR_housing)

    print(housing_dataset[0].shape)

    alpha_c = []
    beta_c = []
    lambda_crime = []

    alpha_h = []
    beta_h = []
    lambda_housing = []

    err_ML_C = []
    err_B_C = []

    err_ML_H = []
    err_B_H = []

    split_list = np.round(np.arange(0.1,1.1,0.1),1)
    for i in split_list:
        print("#######################")
        train_x_C,train_y_C = train_split(crime_dataset[0],crime_dataset[1],i)
        train_x_H, train_y_H = train_split(housing_dataset[0],housing_dataset[1],i)

        a_c, b_c = Model_Selection(train_x_C,train_y_C)
        #print(train_x_H.shape)
        a_h,b_h = Model_Selection(train_x_H,train_y_H)

        l_c = float(a_c/b_c)
        l_h = float(a_h/b_h)

        E_ML_C = MLE(train_x_C,train_y_C,crime_dataset[2],crime_dataset[3], mle_lambda)
        E_MS_C = MLE(train_x_C,train_y_C,crime_dataset[2],crime_dataset[3], l_c)

        E_ML_H = MLE(train_x_H,train_y_H,housing_dataset[2],housing_dataset[3], mle_lambda)
        E_MS_H = MLE(train_x_H,train_y_H,housing_dataset[2],housing_dataset[3], l_h)

        err_ML_C.append(E_ML_C)
        err_B_C.append(E_MS_C)

        err_ML_H.append(E_ML_H)
        err_B_H.append(E_MS_H)

        alpha_c.append(a_c)
        alpha_h.append(a_h)

        beta_c.append(b_c)
        beta_h.append(b_h)

        lambda_crime.append(l_c)
        lambda_housing.append(a_h/b_h)

    ## Plotting curves
    print("Hi")

    fig, ax = plt.subplots(2,2)
    print(ax)
    ax[0][0].plot(split_list,err_ML_C)
    ax[0][0].set_title('MSE_ML_Crime_Dataset')
    ax[0][1].plot(split_list,err_B_C)
    ax[0][1].set_title('MSE_Bayesian_Crime_Dataset')
    ax[1][0].plot(split_list,err_ML_H)
    ax[1][0].set_title('MSE_ML_Housing_Dataset')
    ax[1][1].plot(split_list,err_B_H)
    ax[1][1].set_title('MSE_B_Housing_Dataset')
    fig.tight_layout()

    plt.savefig('Task1_All_' + str(mle_lambda) + '.png')


def task_1():
    ## Run MLE and model_selection with lambda = 0
    mle_lambda = 0
    task_1_helper(mle_lambda)

    mle_lambda = 1.0
    task_1_helper(mle_lambda)

    mle_lambda = 33.0
    task_1_helper(mle_lambda)

    mle_lambda = 100.0
    task_1_helper(mle_lambda)

    mle_lambda = 1000.0
    task_1_helper(mle_lambda)

def task_2():
    data_f3 = load_dataset(train_f3,trainR_f3,test_f3,testR_f3)
    data_f5 = load_dataset(train_f5,trainR_f5,test_f5,testR_f5)

    x_f3 = data_f3[0]
    x_f5 = data_f5[0]

    t_x_f3 = data_f3[2]
    t_x_f5 = data_f5[2]

    f3_rows = data_f3[0].shape[0]
    f5_rows = data_f5[0].shape[0]

    log_evidence = list()
    log_evidence_5 = list()

    error_B = []
    error_ML = []

    error_B_5 = []
    error_ML_5 = []

    for d in range(1,11):
        f3_new = {}
        f3_test_new = {}

        f5_new = {}
        f5_test_new = {}

        for j in range(0,d+1):
            column = np.round(np.power(x_f3,j),2)
            column_test = np.round(np.power(t_x_f3,j),2)

            col_f5 = np.round(np.power(x_f5,j),2)
            colT_f5 = np.round(np.power(t_x_f5,j),2)

            f3_new[j] = list(column[0])
            f3_test_new[j] = list(column_test[0])

            f5_new[j] = list(col_f5[0])
            f5_test_new[j] = list(colT_f5[0])

        f3_dataset = pd.DataFrame(f3_new)
        f3_test_dataset = pd.DataFrame(f3_test_new)

        f5_dataset = pd.DataFrame(f5_new)
        f5_test_dataset = pd.DataFrame(f5_test_new)
        #print(data_f3)

        alpha, beta = Model_Selection(f3_dataset,data_f3[1])
        print(alpha,beta)
        lambda_v = float(alpha/beta)
        log_evid = float(compute_logevidence(d,f3_rows,alpha,beta,f3_dataset,data_f3[1]))
        log_evidence.append(log_evid)

        err_B = MLE(f3_dataset,data_f3[1],f3_test_dataset,data_f3[3], lambda_v)
        err_ML = MLE(f3_dataset, data_f3[1], f3_test_dataset, data_f3[3], 0)

        error_B.append(err_B)
        error_ML.append(err_ML)

        alpha, beta = Model_Selection(f5_dataset,data_f5[1])
        print(alpha,beta)
        lambda_v = float(alpha/beta)
        log_evid = float(compute_logevidence(d,f5_rows,alpha,beta,f5_dataset,data_f5[1]))
        log_evidence_5.append(log_evid)

        err_B = MLE(f5_dataset,data_f5[1],f5_test_dataset,data_f5[3], lambda_v)
        err_ML = MLE(f5_dataset, data_f5[1], f5_test_dataset, data_f5[3], 0)

        error_B_5.append(err_B)
        error_ML_5.append(err_ML)

    dimen = [x for x in range(1,11)]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    ax1.plot(dimen, error_B)
    ax2.plot(dimen, error_ML)
    ax3.plot(dimen, error_B_5)

    ax4.plot(dimen, error_ML_5)

    print(error_ML_5)
    print(error_B_5)

    plt.savefig('Dimensions_Errorqew.png')

    return

if __name__ == '__main__':
    #train_crime, trainR_crime, test_crime, testR_crime = load_dataset(train_crime,trainR_crime,test_crime,testR_crime)

    #task_1()
    task_2()

