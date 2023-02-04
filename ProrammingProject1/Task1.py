#Importing the required Modules.

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

## Creating a global dictionary of words

dictionary_global = {}

train_sizes = ['N/128', 'N/64', 'N/16', 'N/4', 'N']
MLE_train = []
MAP_train = []
Pred_train = []
MLE_test = []
MAP_test = []
Pred_test = []

#Importing train and test data

train_path = './pp1data/training_data.txt'
test_path = './pp1data/test_data.txt'
train_string = ""
test_string = ""
train_set = []
test_set = []

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



def train_split(train_set, split):
    size = int(len(train_set)/ split)
    return train_set[:size]

def MLE(split):
    train = train_split(train_set, split)
    ## Counting frequency of words in the given train split.

    dictionary_tmp = dictionary_global.copy()

    for word in train:
        dictionary_tmp[word]+=1

    ## Compute Probabilities

    prob_dict = {}

    length_train = len(train)

    for (word,freq) in dictionary_tmp.items():
        if(freq!=0):
            prob_dict[word] = math.log(float(freq/length_train))
        else:
            prob_dict[word] = -math.inf
            #prob_dict[word] = math.log(float(0.0001/length_train))

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

    #print("     Maximum Likelihood    ", "    {0}    ", "          {1:6.3f}         "
     #     .format(perplexity_train,perplexity_test))
    print("   N/{0:<3d}      Maximum Likelihood           {1:3.3f}                 {2:6.3f}"
          .format(split,perplexity_train, perplexity_test))

    return perplexity_train, perplexity_test

def MAP(split, alpha):
    train = train_split(train_set, split)
    ## Counting frequency of words in the given train split.

    dictionary_tmp = dictionary_global.copy()

    k_words = []

    ## Checking the count of unique words in the given train set.

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

    denominator = int(length_train + alpha_0 - K)

    for (word,freq) in dictionary_tmp.items():
        numerator = int(freq + alpha_k - 1)
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

    print("   N/{0:<3d}   Maximum A Pesteriori (MAP)      {1:3.3f}                  {2:6.3f}"
          .format(split,perplexity_train, perplexity_test))
    return perplexity_train, perplexity_test


def Predictive_distribution(split, alpha):
    train = train_split(train_set, split)

    #print(type(alpha))
    ## Counting frequency of words in the given train split.

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

    print("   N/{0:<3d}    Predictive Distribution        {1:3.3f}                  {2:6.3f}"
          .format(split,perplexity_train, perplexity_test))
    return perplexity_train, perplexity_test

def pipeline():

    print("*----------------------------------------------------------------------------------*")
    print("*              Change of test perplexities for different dataset sizes             *")
    print("*----------------------------------------------------------------------------------*")
    print("\n")

    ## For Train set size: N/128

    print("************************************************************************************")

    print("\033[1mTrain Size          Model         ", "    Perplexity Train    ", "     Perplexity Test\033[0m    ")
    print("\n")
    p_train, p_test = MLE(128)
    MLE_train.append(p_train)
    MLE_test.append(p_test)

    p_train, p_test = MAP(128,2)
    MAP_train.append(p_train)
    MAP_test.append(p_test)

    p_train, p_test = Predictive_distribution(128,2) # Using Dirichlet Prior-2.
    Pred_train.append(p_train)
    Pred_test.append(p_test)


    ## For Train set size: N/64

    print("\n")


    p_train, p_test = MLE(64)
    MLE_train.append(p_train)
    MLE_test.append(p_test)

    p_train, p_test = MAP(64,2)
    MAP_train.append(p_train)
    MAP_test.append(p_test)

    p_train, p_test = Predictive_distribution(64,2) # Using Dirichlet Prior-2.
    Pred_train.append(p_train)
    Pred_test.append(p_test)

    ## For Train set size: N/16

    print("\n")

    p_train, p_test = MLE(16)
    MLE_train.append(p_train)
    MLE_test.append(p_test)

    p_train, p_test = MAP(16,2)
    MAP_train.append(p_train)
    MAP_test.append(p_test)

    p_train, p_test = Predictive_distribution(16,2) # Using Dirichlet Prior-2.
    Pred_train.append(p_train)
    Pred_test.append(p_test)

    ## For Train set size: N/4

    print("\n")

    p_train, p_test = MLE(4)
    MLE_train.append(p_train)
    MLE_test.append(p_test)

    p_train, p_test = MAP(4,2)
    MAP_train.append(p_train)
    MAP_test.append(p_test)

    p_train, p_test = Predictive_distribution(4,2) # Using Dirichlet Prior-2.
    Pred_train.append(p_train)
    Pred_test.append(p_test)



    ## For Train set size: N

    print("\n")


    p_train, p_test = MLE(1)
    MLE_train.append(p_train)
    MLE_test.append(p_test)

    p_train, p_test = MAP(1,2)
    MAP_train.append(p_train)
    MAP_test.append(p_test)

    p_train, p_test = Predictive_distribution(1,2) # Using Dirichlet Prior-2.
    Pred_train.append(p_train)
    Pred_test.append(p_test)

    print("************************************************************************************")

def plotresults():

    ## Plotting Train perplexities as a function of train dataset size.
    X_axis = np.arange(len(train_sizes))

    plt.figure(figsize=(10, 8))

    plt.bar(X_axis - 0.2, MLE_train, 0.2, label='MLE')
    plt.bar(X_axis + 0.2, MAP_train, 0.2, label='MAP')
    plt.bar(X_axis, Pred_train, 0.2, label='Predictive_distribution')

    plt.xticks(X_axis, train_sizes)
    plt.xlabel("Size of Training dataset", fontsize=15, fontname='Sans')
    plt.ylabel("Perplexity", fontsize=15, fontname='Sans')
    plt.title("Perplexities of train set for different train sizes", fontsize=20, fontname='Sans')
    plt.legend()
    plt.savefig('Train_Perplexities.png')
    plt.show()

    ## Plotting Test perplexities as a function of train dataset size.

    index = 0
    for i in MLE_test:
        if math.isinf(i):
            MLE_test[index] = 15000
        index += 1

    X_axis = np.arange(len(train_sizes))

    plt.figure(figsize=(10, 8))

    plt.bar(X_axis - 0.2, MLE_test, 0.2, label='MLE')
    plt.bar(X_axis + 0.2, MAP_test, 0.2, label='MAP')
    plt.bar(X_axis, Pred_test, 0.2, label='Predictive_distribution')

    plt.axhline(y=15000, color='r', linestyle='--')
    plt.text(-1, 15000, s='~ infinity line', color="red", ha="center", va="center", fontsize='xx-large')

    plt.xticks(X_axis, train_sizes)
    plt.xlabel("Size of Training dataset", fontsize=15, fontname='Sans')
    plt.ylabel("Perplexity", fontsize=15, fontname='Sans')
    plt.title("Perplexities of test set for different train sizes", fontsize=15, fontname='Sans')
    plt.legend()
    plt.savefig("Test_Perplexities.png")
    plt.show()

    return

if __name__=='__main__':
    pipeline()
    plotresults()

