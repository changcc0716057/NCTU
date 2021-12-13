import numpy as np
import pandas as pd
import random
import copy
import math
import time

filename = "glass.data"
epsilon = 10 ** (-6)
nAttribute = -1

# load file and preprocess datas
def load_data(filename, partition=(8,2)):
    # read files
    origin_data = pd.read_csv(filename, sep=",", header=None).to_numpy()
    n_data = origin_data.shape[0]
    
    # shuffle the data
    for i in range(0,10):
        np.random.shuffle(origin_data)

    # calculate where to spilt data
    spiltpoint = round(n_data * partition[0] / (sum(partition)))
    train_x, train_y, test_x, test_y = None, None, None, None

    # we need to spilt different dataset with different way
    if filename == "glass.data":
        train_x = origin_data[0:spiltpoint, 1:-1]
        train_y = origin_data[0:spiltpoint, -1]
        test_x = origin_data[spiltpoint:n_data, 1:-1]
        test_y = origin_data[spiltpoint:n_data, -1]
    elif filename == "iris.data" or filename == "ionosphere":
        train_x = origin_data[0:spiltpoint, 0:-1]
        train_y = origin_data[0:spiltpoint, -1]
        test_x = origin_data[spiltpoint:n_data, 0:-1]
        test_y = origin_data[spiltpoint:n_data, -1]
    else:
        train_x = origin_data[0:spiltpoint, 1:]
        train_y = origin_data[0:spiltpoint, 0]
        test_x = origin_data[spiltpoint:n_data, 1:]
        test_y = origin_data[spiltpoint:n_data, 0]

    return train_x, train_y, test_x, test_y
    
# Compute Gini's impurity
def Gini(labels):
    _ , counts = np.unique(labels, return_counts=True)
    impurity = sum([(count/labels.shape[0]) ** 2 for count in counts])
    return 1 - impurity
    
class Node:
    def __init__(self, depth=0):
        self.FalseNode = None
        self.TrueNode = None
        self.leaf = True
        self.majorlabel = None
        self.attribute = None
        self.threshold = None
        self.depth = depth
        
class CART:
    def __init__(self):
        self.root = Node()

    # Given a threshold, spilt the data into two set by this threshold
    def spiltNode(self, datas, attribute, threshold):
        # FSetId for indices of false set; TsetId for indices of true set
        FSetID = []
        TSetID = []

        # According to the type of data, split the data in different way 
        if isinstance(threshold, int) or isinstance(threshold, float):
            for i in range(0,len(datas[:,attribute])):
                if (datas[:,attribute][i] < threshold):
                    FSetID.append(i)
                else:
                    TSetID.append(i)
        else:
            for i in range(0,len(datas[:,attribute])):
                if (datas[:,attribute][i] != threshold):
                    FSetID.append(i)
                else:
                    TSetID.append(i)
        return FSetID, TSetID

    # recursively build the CART
    # stop when we cannot reduce the impurity
    def buildCART(self, curNode, datas, labels, attrilist, max_depth, \
        min_impurity_decrease=0.0, min_samples_split=2, criterion=Gini):
        """
        max_depth: The maximum depth of the tree
        min_impurity_decrease: A node will be split if this split induces a decrease 
        of the impurity greater than or equal to this value
        min_samples_split: The minimum number of samples required to split an 
        internal node 
        criterion: The function to measure the quality of a split
        """
        
# check if we meet the limit
        if curNode.depth >= max_depth or labels.shape[0] <= min_samples_split:
            key , counts = np.unique(labels, return_counts=True)
            curNode.majorlabel = key[np.argmax(counts)]
            return

        # if the label is unique in this node
        if np.unique(labels).shape[0] == 1:
            curNode.majorlabel = labels[0]
            return

        # initialize 
        parent_imp = criterion(labels)  
        max_impurity_decrease = 0.0
        best_attribute, best_threshold = None, None

        # reset min_impurity_decrease
        min_impurity_decrease = max(min_impurity_decrease, epsilon)        

        # find the best attribute for reducing most impurity
        for curAtt in attrilist:

            # collect all possible threshold
            threslist = np.unique(datas[:,curAtt])
            if isinstance(datas[0][curAtt], int) or \
            isinstance(datas[0][curAtt], float):
                threslist = [(threslist[i-1]+threslist[i])/2 \
                for i in range(1,len(threslist))] 

            # find the best threshold for reducing most impurity
            for threshold in threslist:
                FSetID, TSetID = self.spiltNode(datas, curAtt, threshold)
                ratio = len(FSetID) / labels.shape[0]
                impurity_decrease = parent_imp - \
                    ratio * criterion(labels[FSetID]) - \
                        (1 - ratio) * criterion(labels[TSetID])

                # we get better attribute and threshold to spilt the node
                if impurity_decrease > max_impurity_decrease:
                    max_impurity_decrease = impurity_decrease
                    best_attribute, best_threshold = curAtt, threshold

        # the reduction of impurity is more than limit
        if max_impurity_decrease > min_impurity_decrease:
            Best_FSetID, Best_TSetID = \
                self.spiltNode(datas, best_attribute, best_threshold)
            curNode.FalseNode, curNode.TrueNode = \
                Node(curNode.depth+1), Node(curNode.depth+1)
            curNode.attribute, curNode.threshold = \
                best_attribute, best_threshold
            curNode.leaf = False  # we can spilt more
            
            self.buildCART(curNode.FalseNode, datas[Best_FSetID], \
            labels[Best_FSetID], attrilist, max_depth, min_impurity_decrease, \
            min_samples_split, criterion)
            self.buildCART(curNode.TrueNode, datas[Best_TSetID], \
            labels[Best_TSetID], attrilist, max_depth, min_impurity_decrease, \
            min_samples_split, criterion)
        # the reduction cannot meet the limit, 
        # so this node is leaf and should compute majorlabel
        else:
            key , counts = np.unique(labels, return_counts=True)
            curNode.majorlabel = key[np.argmax(counts)]

    def train(self, datas, labels, max_depth=None, max_features="auto", \
    bootstrap=True, min_impurity_decrease=0.0, min_samples_split=2, criterion=Gini):
        if max_depth == None:
            max_depth = 10 ** 9

        # Attribute Bagging
        if bootstrap == True:
            attributeID = \
            np.random.choice(a=nAttribute, size=max_features, replace=False)
        else:
            attributeID = [i for i in range(0, nAttribute)]
        self.buildCART(self.root, datas, labels, attributeID, max_depth, \
             min_impurity_decrease, min_samples_split, criterion)

    def predict(self, testcase):
        curNode = self.root
        while (not curNode.leaf):
            if isinstance(curNode.threshold, int) or \
            isinstance(curNode.threshold, float):
                if testcase[curNode.attribute] < curNode.threshold:
                    curNode = curNode.FalseNode
                else:
                    curNode = curNode.TrueNode
            else:
                if testcase[curNode.attribute] == curNode.threshold:
                    curNode = curNode.FalseNode
                else:
                    curNode = curNode.TrueNode
        return curNode.majorlabel

    def cal_accuracy(self, testcases, labels):
        ncase = len(labels)
        correct = \
        [int(self.predict(testcases[i]) == labels[i]) for i in range(0,ncase)]
        
        return float(sum(correct)) / ncase
        
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features="auto", \
    max_samples=None, bootstrap=True, min_impurity_decrease=0.0, \
    min_samples_split=2, criterion=Gini):
        self.CARTrees = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def train(self, datas, labels):
        # According to different input, give the corresponding number of features
        if self.max_features == "auto" or self.max_features == "sqrt":
            self.max_features = math.sqrt(nAttribute)
        elif self.max_features == "log2":
            self.max_features = math.log2(nAttribute)
        elif type(self.max_features) == float:
            self.max_features = self.max_features * nAttribute
        elif self.max_features == None:
            self.max_features = nAttribute

        if self.max_samples == None:
            self.max_samples = datas.shape[0]
        elif type(self.max_samples) == float:
            self.max_samples = self.max_samples * datas.shape[0]

        # ensure that the number of  samples and features is less than 
        # or equal to the original size
        self.max_samples = min(round(self.max_samples), datas.shape[0])
        self.max_features = min(round(self.max_features), nAttribute)

        for i in range(0, self.n_estimators):
            tree = CART()
            if self.bootstrap == True:
                sampleID = np.random.choice(a=datas.shape[0], \
                size=self.max_samples, replace=False)
            else:
                sampleID = [i for i in range(0, datas.shape[0])]

            tree.train(datas[sampleID], labels[sampleID], self.max_depth, \
            self.max_features, self.bootstrap, self.min_impurity_decrease, \
            self.min_samples_split, self.criterion)
            self.CARTrees.append(tree)

    def predict(self, testcase):
        predict_labels = \
        np.array([tree.predict(testcase) for tree in self.CARTrees])
        key , counts = np.unique(predict_labels, return_counts=True)
        return key[np.argmax(counts)]

    def cal_accuracy(self, testcases, labels):
        ncase = len(labels)
        correct = \
        [int(self.predict(testcases[i]) == labels[i]) for i in range(0,ncase)]
        return float(sum(correct)) / ncase 

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data(filename)
    nAttribute = train_x.shape[1]
    attrilist = [i for i in range(0, nAttribute)]

    a = time.time()
    RF = RandomForest(n_estimators=1, max_depth=12, max_features="sqrt", \
    max_samples=100, bootstrap=True, min_impurity_decrease=epsilon, \
    min_samples_split=2)
    RF.train(train_x, train_y)
    train_accuracy = RF.cal_accuracy(train_x, train_y)
    test_accuracy = RF.cal_accuracy(test_x, test_y)
    b = time.time()
    print("Train Accuracy:", train_accuracy, '\n', end='')
    print("Test Accuracy:", test_accuracy, '\n', end='')
    print("Used Time:", b-a, "s")