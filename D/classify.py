# SeekTruth.py : Classify text objects into two categories
#
# PLEASE PUT YOUR NAMES AND USER IDs HERE
# Netaji Sai Pavan Neerukonda - neneer@iu.edu
# Sriram Reddy Pidaparthi - sripidap@iu.edu
# Venkata Dinesh Gopu - vgopu@iu.edu
#
# Based on skeleton code by D. Crandall, October 2021
#
import sys
def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")
    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
def classifier(train_data, test_data):
    N_positive = 0
    K = 2
    alpha = 0.5
    # This is just dummy code -- put yours here!
    stopwords = ["ourselves", "between", "yourself",  "once", "they", "own", "an", "be", "for", "do", "yours", "into", "of", "most", "itself", "is", "s", "am", "or",
                 "themselves", "below", "are", "we", "these", "your", "through",  "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their",
                 "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "them", "same", "and", "been", "have", "in", "will", "on", "does",
                 "yourselves", "then", "that", "what", "over", "why", "so", "did", "not", "now", "under", "you", "herself", "has", "just", "where", "too", "only", "myself", "which",
                 "those", "i", "after", "whom", "being", "theirs", "my", "a", "by", "doing", "it", "how", "was", "here", "than"]
    bagDict,updatedList= {},[]
    for i in range(len(train_data["objects"])):
        for term in train_data["objects"][i].split():
            if term in stopwords:
                continue
            if term not in bagDict:
                bagDict[term] = dict()
            if train_data["labels"][i] == "truthful":
                if 'truthful' not in bagDict[term]:
                    bagDict[term]['truthful'] = 0
                bagDict[term]["truthful"] += 1
            else:
                if 'deceptive' not in bagDict[term]:
                    bagDict[term]['deceptive'] = 0
                bagDict[term]["deceptive"] += 1
        if(train_data['labels'][i] == "deceptive"):
            N_positive+=1
    for i in range(len(test_data['objects'])):
        threshold = 1
        for term in test_data['objects'][i].split():
            if term in bagDict and 'truthful' in bagDict[term] and 'deceptive' in bagDict[term]:
                div = bagDict[term]['deceptive']/bagDict[term]['truthful']
                #div = div/(N_positive + K * alpha)
                threshold = threshold / div
                #word_occ = (bagDict[term]['deceptive']/bagDict[term]['truthful']) + alpha
            else:
                #threshold /= 0.5
                continue
                div = alpha
                word_occ = div/(N_positive+K*alpha)
                threshold/=word_occ
            #word_occ = div / (N_positive + K * alpha)
            #threshold = threshold/div
            #threshold = threshold/word_occ
        if (threshold <= 1):
            updatedList.append("deceptive")
        else:
            updatedList.append("truthful")
    return updatedList


if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    raise Exception("Usage: classify.py train_file.txt test_file.txt")
    train_file = './train_file.txt'
    test_file = './test_file.txt'
    #(_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))