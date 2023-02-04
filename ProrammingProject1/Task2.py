#Importing the required Modules.

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

## Creating a global dictionary of words

dictionary_global = {}
log_evidences = []
perplexities_test = []
train_set = []
test_set = []
alpha_scalar = [i for i in range(1,11)]

#Importing train and test data

train_path = './pp1data/training_data.txt'
test_path = './pp1data/test_data.txt'
train_string = ""
test_string = ""

with open(train_path, 'r') as train:
    train_string = train.read()

with open(test_path, 'r') as test:
    test_string = test.read()

train_set = train_string.split(' ')
test_set = test_string.split(' ')

for i in train_set:
    if i not in dictionary_global.keys():
        dictionary_global[i] = 0

for i in test_set:
    if i not in dictionary_global.keys():
        dictionary_global[i] = 0

def plotresults():
    # plt.figure(figsize = (10,8))
    fig, axis = plt.subplots(1, 2, figsize=(20, 8))

    axis[1].plot(alpha_scalar, perplexities_test, label='test perplexities')

    axis[1].set(xlabel="Alpha values", ylabel="Test Perplexities",
                title="Perplexities of test set for different alpha values for Predictive Distribution (N/128)")

    # plt.plot(alpha_scalar, perplexities_test, label = 'test perplexities')

    axis[0].plot(alpha_scalar, log_evidences, label='log evidences')
    axis[0].set(xlabel='Alpha values', ylabel="Log Evidences",
                title='Log Evidence values for different alpha values')
    fig.savefig("Task2.png")
    # axis[1].savefig("TestP.png")
    return


def train_split(train_set, split):
    size = int(len(train_set)/ split)
    return train_set[:size]

def compute_evidence(split, alpha):

    train = train_split(train_set, split)
    ## Counting frequency of words in the given train split.

    dictionary_tmp = dictionary_global.copy()

    N = len(train)

    k_words = []

    for i in train:
        if i not in k_words:
            k_words.append(i)

    K = len(k_words)


    for word in train:
        dictionary_tmp[word]+=1

    #print(K)

    alpha_k = alpha
    alpha_0 = alpha_k*K

    term1 = math.log(math.factorial(alpha_0 - 1))

    term2 = 0.0
    cnt = 0
    for word in k_words:
        term2+= math.factorial(dictionary_tmp[word] + alpha_k - 1)
        #print(dictionary_tmp[word])
        cnt+=1

    term3 = math.log(math.factorial(alpha_0 + N - 1))

    term4 = K*math.factorial(alpha_k-1)

    log_evidence = term1 + term2 - term3 - term4

    return log_evidence

def Predictive_distribution(split, alpha):
    train = train_split(train_set, split)

    dictionary_tmp = dictionary_global.copy()


    for word in train:
        dictionary_tmp[word]+=1

    k_words = []

    for i in train:
        if i not in k_words:
            k_words.append(i)

    K = len(k_words)

    alpha_k = int(alpha)
    alpha_0 = alpha_k*K

    for word in train:
        dictionary_tmp[word]+=1

    ## Compute Probabilities

    prob_dict = {}

    length_train = len(train)

    denominator = int(length_train + alpha_0)

    for (word,freq) in dictionary_tmp.items():
        numerator = int(freq + alpha_k)
        prob_dict[word] = math.log(float(numerator/denominator))

    perplexity_train, perplexity_test = 0.0,0.0

    for word in train:
        perplexity_train+=prob_dict[word]

    perplexity_train = float(perplexity_train/length_train)
    perplexity_train = math.exp(-perplexity_train)

    ## Calculating perplexity for test_set

    for word in test_set:
        perplexity_test+=prob_dict[word]

    length_test = len(test_set)
    perplexity_test = float(perplexity_test/length_test)
    perplexity_test = math.exp(-perplexity_test)

   # print("   N/{0:<3d}    Predictive Distribution        {1:3.3f}                  {2:6.3f}"
   #       .format(split,perplexity_train, perplexity_test))
    return perplexity_train, perplexity_test

def model_selection():

    print("*----------------------------------------------------------------------------------*")
    print("*      Log evidence and test perplexities for different values of alpha prior      *")
    print("*----------------------------------------------------------------------------------*")
    print("\n")

    print("************************************************************************************")

    print("\033[1mAlpha Value               Log Evidence             Perplexity Test\033[0m    ")
    print("\n")
    for i in range(1,11):
        log_evidence = compute_evidence(128, i)
        log_evidences.append(log_evidence)
        p_train, p_test = Predictive_distribution(128, i)
        perplexities_test.append(p_test)
        print("    {0:<2d}                 {1:^20}          {2}".format(i,round(log_evidence,2), round(p_test,2)))
    print("************************************************************************************")

if __name__=='__main__':
    model_selection()
    plotresults()




