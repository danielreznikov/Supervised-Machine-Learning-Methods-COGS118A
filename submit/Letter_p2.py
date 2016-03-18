
# coding: utf-8

# In[1]:

# Copyright Daniel Reznikov, Yunfan Yang all rights reserved
# Final Project COGS 118a, Winter 2016
# Notebook Implements K-Nearest-Neighbors Classifier on breast_cancer dataset


# In[1]:

import warnings
warnings.filterwarnings("ignore")
import sklearn.cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np

# Import data
# X: features is a vectors of 16 integer attributes extracted from raster scan images of the letters.
X = [list(map(int, x.split(',')[1:])) for x in open('letter-recognition.data').read().splitlines()]

# Y: the label, we are setting label = 1 when letter = 'O'
Y = [1 if x.split(',')[0] <= 'M' else -1 for x in open('letter-recognition.data').read().splitlines()]

# Split the dataset. Training will be on 5000/20,000 as in the Caruana 06' Paper
xTrain, xTest, yTrain, yTest = cv.train_test_split(X, Y, train_size = 5000/float(len(X)))


# In[2]:

import Classifiers as clfs
clfs.KNN(xTrain, xTest, yTrain, yTest)
clfs.RandomForest(xTrain, xTest, yTrain, yTest)
#clfs.SVM(xTrain, xTest, yTrain, yTest)
clfs.linearSVC(xTrain, xTest, yTrain, yTest)
clfs.BoostedDecisionTree(xTrain, xTest, yTrain, yTest)
clfs.NeuralNets(xTrain, xTest, yTrain, yTest)
clfs.XGBoost(xTrain, xTest, yTrain, yTest)


# In[ ]:



