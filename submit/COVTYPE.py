
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from collections import Counter

# cap input data size
SIZE_DATA = 30000

def largestClass(labels):
    c = Counter(labels)
    return c.most_common()[0][0]

# import data
# X: 54 features
# Y: 1 label 
X = [list(map(int, x.split(',')[:-1])) for x in open('covtype.data').read().splitlines()[:SIZE_DATA]]
_Y = [x.split(',')[-1] for x in open('covtype.data').read().splitlines()[:SIZE_DATA]]
larg = largestClass(_Y)
# treat the largest class as positive, the rest as negative
Y = [1 if x == larg else -1 for x in _Y]

xTrain, xTest, yTrain, yTest = cv.train_test_split(X, Y, train_size = 5000/len(X))


# In[2]:

import Classifiers as clfs
clfs.KNN(xTrain, xTest, yTrain, yTest)
clfs.RandomForest(xTrain, xTest, yTrain, yTest)
clfs.BoostedDecisionTree(xTrain, xTest, yTrain, yTest)
clfs.NeuralNets(xTrain, xTest, yTrain, yTest)
#clfs.SVM(xTrain, xTest, yTrain, yTest)
clfs.linearSVC(xTrain, xTest, yTrain, yTest)
import Classifiers as clfs
clfs.XGBoost(xTrain, xTest, yTrain, yTest)


# In[ ]:



