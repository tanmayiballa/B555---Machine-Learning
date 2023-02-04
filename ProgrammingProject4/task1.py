import random

import numpy as np
import pandas as pd

artificial = './pp4data/artificial/'
newsgroups = './pp4data/20newsgroups/'
task_1_fm = []
art_D = 10
news_D = 200

def Gibbs_Sampler(K,alpha,beta,N,words,documents,topics,D,dataset):
    ## Construct Cd matrix D*K (Documents * Topics)
    CD = [[0 for i in range(K)] for j in range(len(D))]
    CT = [[0 for i in range(K)] for j in range(len(words))]
    ind = 0
    prob = np.array([0.0 for i in range(K)])
    for i in D:
        with open(dataset + str(i)) as doc:
            text = doc.readlines()[0]
            for word in text.split(' '):
                CD[i-1][topics[ind]]+=1
                ind+=1
    #print(CD)
    word_topic_dict = {}
    #beta = {}
    for i in range(len(words)):
        word = words[i]
        topic = topics[i]
        #print(word)
        if word in word_topic_dict.keys():
            word_topic_dict[word][topic]+=1
        else:
            word_topic_dict[word] = [0 for i in range(K)]
            word_topic_dict[word][topic] += 1
    CT = word_topic_dict.copy()
    topic_choices = list(range(0,K))

    for i in range(N):
        print("Iteration: ",i)
        phi_n = np.random.permutation(len(words))
        for l in range(len(words)):
            j = phi_n[l]
            #print(j)
            word = words[j]
            topic = topics[j]
            document = documents[j]-1
            CD[document][topic]-=1
            CT[word][topic]-=1
            V = len(CT)

            for k in range(K):
                sum_ct = 0
                for key, val in CT.items():
                    sum_ct += val[k]
                sum_cd = CD[document][0] + CD[document][1]
                prob[k] = ((CT[word][k] + beta)*(CD[document][k] + alpha))/((V*beta + sum_ct) * (K*alpha + sum_cd))
            prob_nrm = prob/np.linalg.norm(prob)

            new_topic = random.choices(topic_choices,prob_nrm)[0]
            #print(new_topic)
            CD[document][new_topic]+=1
            CT[word][new_topic]+=1
            topics[j] = new_topic
            #print(CT)

    # Output 5 most frequent words for a topic:
    topic_dict = {}
    for i in range(K):
        topic_dict[i] = []
    for i in CT.keys():
        for j in range(K):
            topic_dict[j].append([CT[i][j],i])
    res = []
    #print("Topics dict")
    #print(topic_dict)
    for i in topic_dict.keys():
        words_topic = topic_dict[i]
        #print(words_topic)
        words_topic.sort(reverse = True)
        #print(words_topic)
        line = ""
        #print(words_topic)
        for k in range(5):
            line=line+words_topic[k][1]+" "
        res.append(line)
    df = pd.DataFrame(res)
    df.to_csv('topicwords.csv',index = False, header = False)

    ## Creating the feature vector
    D_feature = []
    for i in D:
        doc_fm = []
        for j in range(K):
            #print(CD[i-1],"cd")
            doc_fm.append((CD[i-1][j] + alpha)/(K*alpha + sum(CD[i-1])))
            #print(sum(CD[i-1]),"cdsum")
        D_feature.append(doc_fm)
    task_1_fm = D_feature
    nparr = np.array(task_1_fm)
    np.save('f1.npy', nparr)
    #print(nparr)
    return

def task1_helper(dataset,K,d):
    D = list(range(1,d+1))
    topic_choices = list(range(0,K))

    words = []
    topics = []
    documents = []
    for i in D:
        with open(dataset + str(i)) as doc:
            text = doc.readlines()[0]
            for word in text.split(' '):
                words.append(word)
                t = np.random.choice(topic_choices)
                topics.append(t)
                documents.append(i)
    Gibbs_Sampler(K,5/K,0.01,500,words,documents,topics,D,dataset)


if __name__ == '__main__':
    task1_helper(newsgroups, 20, 200)