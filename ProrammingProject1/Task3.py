import pandas as pd
import math


def train_split(train_set, split):
    size = int(len(train_set)/ split)
    return train_set[:size]


train_pathA = './pp1data/pg121.txt.clean'
test_pathA1 = './pp1data/pg141.txt.clean'
test_pathA2 = './pp1data/pg1400.txt.clean'

train_setA = []
test_set1A = []
test_set2A = []


with open(train_pathA, 'r') as train:
    for line in train:
        for word in line.split():
            train_setA.append(word)

with open(test_pathA1, 'r') as test:
    for line in test:
        for word in line.split():
            test_set1A.append(word)

with open(test_pathA2, 'r') as test:
    for line in test:
        for word in line.split():
            test_set2A.append(word)

## Creating a dictionary for authors

Adict = {}

for i in train_setA:
    Adict[i] = 0

for i in test_set1A:
    Adict[i] = 0

for i in test_set2A:
    Adict[i] = 0

## Creating a list of unique elements

unique_list = []

for i in train_setA:
    if i not in unique_list:
        unique_list.append(i)

for i in test_set1A:
    if i not in unique_list:
        unique_list.append(i)

for i in test_set2A:
    if i not in unique_list:
        unique_list.append(i)


def Predictive_distribution_Author(split, alpha):
    train = train_split(train_setA, split)

    ## Counting frequency of words in the given train split.

    dictionary_tmp = Adict.copy()

    for word in train:
        dictionary_tmp[word]+=1

    k_words = []

    for i in train:
        if i not in k_words:
            k_words.append(i)

    K = len(k_words)

    #print(K)

    alpha_k = int(alpha)
    alpha_0 = alpha_k*K

    for word in train:
        dictionary_tmp[word]+=1

    ## Compute Probabilities

    prob_dict = {}

    length_train = len(train)
    #print(length_train)

    denominator = int(length_train + alpha_0)

    for (word,freq) in dictionary_tmp.items():
        numerator = int(freq + alpha_k)
        prob_dict[word] = math.log(float(numerator/denominator))

    perplexity_train, perplexity_test1, perplexity_test2 = 0.0,0.0,0.0

    for word in train:
        perplexity_train+=prob_dict[word]
    #print(perplexity_train)

    perplexity_train = float(perplexity_train/length_train)
    #print(perplexity_train)
    perplexity_train = math.exp(-perplexity_train)

    ## Calculating perplexity for test_set

    for word in test_set1A:
        perplexity_test1+=prob_dict[word]

    for word in test_set2A:
        perplexity_test2+=prob_dict[word]

    length_test1 = len(test_set1A)
    perplexity_test1 = float(perplexity_test1/length_test1)
    perplexity_test1 = math.exp(-perplexity_test1)

    length_test2 = len(test_set2A)
    perplexity_test2 = float(perplexity_test2/length_test2)
    perplexity_test2 = math.exp(-perplexity_test2)

    print("*--------------------------------------------------------------------------------*")
    print("*    Author Identification using Predictive Distribution with alpha_prior = 2    *")
    print("*--------------------------------------------------------------------------------*")
    print("\n")

    #print("**********************************************************************************")

    print("\033[1mPerplexity_train121:\033[0m {0:3.3f}\n\033[1mPerplexity_test141:\033[0m {1:6.3f}\n\033[1mPerplexity_test1400:\033[0m {2:6.3f}"
          .format(perplexity_train, perplexity_test1, perplexity_test2))

if __name__=='__main__':
    Predictive_distribution_Author(1,2)
