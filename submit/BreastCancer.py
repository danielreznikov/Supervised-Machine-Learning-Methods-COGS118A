
# coding: utf-8

# In[ ]:

# Copyright Daniel Reznikov, Yunfan Yang all rights reserved
# Final Project COGS 118a, Winter 2016
# Notebook Implements SVM Classifier on breast_cancer dataset


# In[1]:

import warnings
warnings.filterwarnings("ignore")
import csv
import numpy as np
from numpy import genfromtxt
import math
import random
import scipy.io as sio
from sklearn.grid_search import GridSearchCV
from sklearn import svm as svm
import sklearn.cross_validation as cv

X = [list(map(int, x.split(',')[:-1])) for x in open('breast_cancer.csv').read().splitlines() if '?' not in x]
Y = [1 if x.split(',')[-1] == '2' else -1 for x in open('breast_cancer.csv').read().splitlines() if '?' not in x]
xTrain, xTest, yTrain, yTest = cv.train_test_split(X, Y, train_size = 0.2)


# In[2]:

import Classifiers as clfs
clfs.KNN(xTrain, xTest, yTrain, yTest)
clfs.RandomForest(xTrain, xTest, yTrain, yTest)
clfs.BoostedDecisionTree(xTrain, xTest, yTrain, yTest)
clfs.NeuralNets(xTrain, xTest, yTrain, yTest)
#clfs.SVM(xTrain, xTest, yTrain, yTest)
clfs.linearSVC(xTrain, xTest, yTrain, yTest)
clfs.XGBoost(xTrain, xTest, yTrain, yTest)


# In[ ]:



