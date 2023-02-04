## Task-1

import os
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import random
import math
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

data_folder = os.path.join(os.getcwd(), 'pp2data')

train_crime = '/train-crime.csv'
trainR_crime = '/trainR-crime.csv'
test_crime = '/test-crime.csv'
testR_crime = '/testR-crime.csv'

train_housing = '/train-housing.csv'
trainR_housing = '/trainR-housing.csv'
test_housing = '/test-housing.csv'
testR_housing = '/testR-housing.csv'

def train_split(train_x,train_y,split_value):
    len = train_x.shape[0]
    train_split = int(split_value*len)
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

def calculate_weights(data, target, lambda_val):
    TCA = calc_piTpi(data)

    identity = np.identity(TCA.shape[0])

    lambda_matrix = lambda_val*identity

    A_inv = lambda_matrix + TCA

    A = np.linalg.inv(A_inv)

    B = np.dot(A,np.transpose(data))

    weights = np.dot(B,target)

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

def LinearRegression(train_x, train_y,test_x,test_y, lambda_val):
    w = calculate_weights(train_x, train_y, lambda_val)
    y_pred = predict_y(test_x, w)
    err = find_mse(y_pred, test_y)
    return err

def Model_Selection(train_x, train_y):
    alpha = random.randint(1, 10)
    beta = random.randint(1, 10)

    it = 0
    N = train_x.shape[0]

    new_alpha = alpha
    new_beta = beta

    reg = math.pow(10,-7)

    while(True):

        BpiTpi = np.dot((beta * np.transpose(train_x)), train_x)
        eigen_val, eigen_vec = np.linalg.eig(BpiTpi)

        alpha_I = alpha * np.identity(BpiTpi.shape[0])
        S_N_inv = alpha_I + BpiTpi
        reg_iden = (reg* np.identity(S_N_inv.shape[0]))
        S_N = np.linalg.inv(S_N_inv+reg_iden)

        bSnPT = np.dot((beta * S_N), np.transpose(train_x))

        m_N = np.dot(bSnPT, train_y)

        # Calc. Gamma

        gamma = np.sum(eigen_val/(eigen_val+alpha))

        # New Alpha

        new_alpha = gamma/np.square(np.linalg.norm(m_N))

        # New Beta

        new_beta = (N - gamma)/(np.square(np.linalg.norm(np.dot(train_x, m_N) - train_y)))

        if(abs(new_alpha - alpha)<=0.0001 and abs(new_beta - beta)<=0.0001):
            break

        alpha = new_alpha
        beta = new_beta

        it+=1

    return np.round(float(new_alpha),2), np.round(float(new_beta),2)

def task_1_helper(mle_lambda):
    crime_dataset = load_dataset(train_crime, trainR_crime, test_crime, testR_crime)
    housing_dataset = load_dataset(train_housing,trainR_housing,test_housing,testR_housing)

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
        train_x_C,train_y_C = train_split(crime_dataset[0],crime_dataset[1],i)
        train_x_H, train_y_H = train_split(housing_dataset[0],housing_dataset[1],i)

        a_c, b_c = Model_Selection(train_x_C,train_y_C)
        a_h,b_h = Model_Selection(train_x_H,train_y_H)

        l_c = np.round(float(a_c/b_c),2)
        l_h = np.round(float(a_h/b_h),2)

        E_ML_C = LinearRegression(train_x_C,train_y_C,crime_dataset[2],crime_dataset[3], mle_lambda)
        E_MS_C = LinearRegression(train_x_C,train_y_C,crime_dataset[2],crime_dataset[3], l_c)

        E_ML_H = LinearRegression(train_x_H,train_y_H,housing_dataset[2],housing_dataset[3], mle_lambda)
        E_MS_H = LinearRegression(train_x_H,train_y_H,housing_dataset[2],housing_dataset[3], l_h)

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

    fig, ax = plt.subplots(2,2)
    #print(ax)
    ax[0][0].plot(split_list,err_ML_C)
    ax[0][0].set_title('MSE_MLE_Crime_Dataset')
    ax[0][1].plot(split_list,err_B_C)
    ax[0][1].set_title('MSE_Bayesian_Crime_Dataset')
    ax[1][0].plot(split_list,err_ML_H)
    if(mle_lambda == 0):
        ax[1][0].set_ylim(0,1.1)
    ax[1][0].set_title('MSE_MLE_Housing_Dataset')
    ax[1][1].plot(split_list,err_B_H)
    ax[1][1].set_title('MSE_Bayesian_Housing_Dataset')
    fig.tight_layout()

    if(mle_lambda == 0):
        plt.savefig('Task_1b.png')
    else:
        plt.savefig('Task1c_' + str(mle_lambda) + '.png')

    return alpha_c, beta_c, lambda_crime, alpha_h, beta_h, lambda_housing

def task_1():

    ### Task-1 (i & ii)
    mle_lambda = 0
    alpha_c, beta_c, lambda_crime, alpha_h, beta_h, lambda_housing = task_1_helper(mle_lambda)
    split_list = np.round(np.arange(0.1,1.1,0.1),1)

    result_dict = {}
    result_dict["Split"] = split_list
    result_dict["Alpha_Crime"] = alpha_c
    result_dict["Beta_Crime"] = beta_c
    result_dict["Lambda_Crime"] = lambda_crime

    result_dict["Alpha_Housing"] = alpha_h
    result_dict["Beta_Housing"] = beta_h
    result_dict["Lambda_Housing"] = lambda_housing

    result_df = pd.DataFrame(result_dict)
    print(tabulate(result_df, headers=result_df.columns, tablefmt='grid'))

    ### Task-1 (iii)
    mle_lambda = 1.0
    task_1_helper(mle_lambda)

    mle_lambda = 33.0
    task_1_helper(mle_lambda)

    mle_lambda = 100.0
    task_1_helper(mle_lambda)

    mle_lambda = 1000.0
    task_1_helper(mle_lambda)

    return

if __name__ == '__main__':

    task_1()