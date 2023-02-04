## Task-1

import os
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from tabulate import tabulate
import random
import math
import warnings
warnings.filterwarnings("ignore")

data_folder = os.path.join(os.getcwd(), 'pp2data')

## Datasets for Task-2

train_f3 = '/train-f3.csv'
train_f5 = '/train-f5.csv'

trainR_f3 = '/trainR-f3.csv'
trainR_f5 = '/trainR-f5.csv'

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

def calculate_weights(data, target, lambda_val):
    TCA = calc_piTpi(data)

    identity = np.identity(TCA.shape[0])
    #print(TCA.shape)
    #print(identity)
    lambda_matrix = lambda_val*identity

    #print("Lambda_Matrix", lambda_matrix.shape, lambda_matrix)
    #print(TCA)

    #tca = np.dot(np.linalg.inv(np.dot(np.transpose(data),data)),(np.dot(np.transpose(data),target)))

    A_inv = lambda_matrix + TCA

    A = np.linalg.inv(A_inv)

    B = np.dot(A,np.transpose(data))
    #print(B.shape)

    weights = np.dot(B,target)
    #print(weights.shape, "Weights")
    #print(weights)
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
    #print(train_x)
    #print(train_y)
    alpha = random.randint(1, 10)
    beta = random.randint(1, 10)

    it = 0
    N = train_x.shape[0]

    new_alpha = alpha
    new_beta = beta

    reg = math.pow(10,-7)

    while(True):

        # Calc S_N

        BpiTpi = np.dot((beta * np.transpose(train_x)), train_x)
        eigen_val, eigen_vec = np.linalg.eig(BpiTpi)

        alpha_I = alpha * np.identity(BpiTpi.shape[0])
        S_N_inv = alpha_I + BpiTpi
        reg_iden = (reg* np.identity(S_N_inv.shape[0]))
        S_N = np.linalg.inv(S_N_inv+reg_iden)

        # Calc m_N

        bSnPT = np.dot((beta * S_N), np.transpose(train_x))

        m_N = np.dot(bSnPT, train_y)

        # Calc. Gamma

        gamma = np.sum(eigen_val/(eigen_val+alpha))

        #print(gamma)

        # New Alpha

        new_alpha = gamma/np.square(np.linalg.norm(m_N))

        # New Beta

        new_beta = (N - gamma)/(np.square(np.linalg.norm(np.dot(train_x, m_N) - train_y)))

        #print(new_alpha, new_beta)
        if(abs(new_alpha - alpha)<=0.0001 and abs(new_beta - beta)<=0.0001):
            #print('End of iteration')
            break

        alpha = new_alpha
        beta = new_beta

        it+=1

    #print("End of iteration")
    return new_alpha, new_beta

def compute_logevidence(M,N,alpha,beta,train_x,train_y):
    term1 = ((M/2)*math.log(alpha))

    term2 = ((N/2)*math.log(beta))

    TCA = calc_piTpi(train_x)

    identity = np.identity(TCA.shape[0])
    lambda_matrix = alpha * identity

    A = lambda_matrix + beta * TCA
    A_inv = np.linalg.inv(A)

    m_N = np.dot(np.dot((beta * A_inv), np.transpose(train_x)), train_y)

    term3 = ((beta/2)*np.square((np.linalg.norm(train_y - (np.dot(train_x,m_N))))) + (alpha/2)*calc_piTpi(m_N))

    term4 = (0.5*math.log(np.linalg.det(A)))


    term5 = ((N/2)*math.log(2*3.14))

    L_evid = term1 + term2 - term3 - term4 - term5

    return L_evid

def task_2_helper(train_x, train_y, test_x, test_y):
    data = load_dataset(train_x, train_y, test_x, test_y)

    x_tr = data[0]
    x_te = data[2]

    N = data[0].shape[0]

    log_evidence = list()

    error_B = []
    error_ML = []

    for d in range(1,11):
        create_trainD = {}
        create_testD = {}

        for j in range(0,d+1):
            column = np.round(np.power(x_tr,j),2)
            column_test = np.round(np.power(x_te,j),2)

            create_trainD[j] = list(column[0])
            create_testD[j] = list(column_test[0])

        train_dataset = pd.DataFrame(create_trainD)
        test_dataset = pd.DataFrame(create_testD)

        ## Calculate alpha, beta from model selection
        alpha, beta = Model_Selection(train_dataset,data[1])

        lambda_v = float(alpha/beta)
        log_evid = float((compute_logevidence(d,N,alpha,beta,train_dataset,data[1]))[0][0])
        log_evidence.append(log_evid)

        err_B = LinearRegression(train_dataset,data[1],test_dataset,data[3], lambda_v)
        err_ML = LinearRegression(train_dataset, data[1], test_dataset, data[3], 0)

        error_B.append(err_B)
        error_ML.append(err_ML)

    return np.round(error_B,2), np.round(error_ML,2), np.round(log_evidence,2)

def task_2():

    dimensions = [x for x in range(1, 11)]
    # Calculate log evidence, MSE for f-3 dataset

    error_B_F3, error_ML_F3, log_evidence_F3 = task_2_helper(train_f3, trainR_f3, test_f3, testR_f3)

    # f-5 dataset

    error_B_F5, error_ML_F5, log_evidence_F5 = task_2_helper(train_f5, trainR_f5, test_f5, testR_f5)

    # Print results

    result_dict = {}
    result_dict["Dimensions"] = dimensions
    result_dict["MSE_ML_F3"] = list(error_ML_F3)
    result_dict["MSE_Bayesian_F3"] = list(error_B_F3)
    result_dict["Log Evidence - F3"] = list(log_evidence_F3)
    result_dict["MSE_ML_F5"] = list(error_ML_F5)
    result_dict["MSE_Bayesian_F5"] = list(error_B_F5)
    result_dict["Log Evidence - F5"] = list(log_evidence_F5)

    result_df = pd.DataFrame(result_dict)
    print(tabulate(result_df, headers=result_df.columns, tablefmt = 'grid'))

    # Plot results
    plt.rcParams.update({'font.size': 20})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (25,10))

    ax1.plot(dimensions, error_ML_F3)
    ax1.set_ylim(0, 9*(10**5))
    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.set_title("MLE - F-3 Dataset")
    ax2.plot(dimensions, error_B_F3)
    ax2.set_ylim(0, 10 ** 6)
    ax2.set_xlabel("Dimensions")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    ax2.set_title("Bayesian - F-3 Dataset")
    ax3.plot(dimensions, log_evidence_F3)
    ax3.set_xlabel("Dimensions")
    ax3.set_ylabel("log Evidence")
    ax3.set_title("Log Evidence Plot")

    plt.savefig('Task-2_F3.png')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (25,10))
    ax1.plot(dimensions, error_ML_F5)
    ax1.set_ylim(0, 10 ** 6)
    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.set_title("MLE - F-5 Dataset")
    ax2.plot(dimensions, error_B_F5)
    ax2.set_ylim(0, 10 ** 6)
    ax2.set_xlabel("Dimensions")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    ax2.set_title("Bayesian - F-5 Dataset")
    ax3.plot(dimensions, log_evidence_F5)
    ax3.set_xlabel("Dimensions")
    ax3.set_ylabel("log Evidence")
    ax3.set_title("Log Evidence Plot")

    plt.savefig('Task-2_F5.png')

    return

if __name__ == '__main__':
    task_2()