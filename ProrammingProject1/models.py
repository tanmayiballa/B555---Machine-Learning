

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