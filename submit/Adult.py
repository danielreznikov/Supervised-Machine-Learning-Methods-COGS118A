
# coding: utf-8

# In[2]:

import warnings
warnings.filterwarnings("ignore")
import sklearn.cross_validation as cv
from collections import Counter
from sklearn import preprocessing
import numpy as np

# Import Data

# cap input data size
SIZE_DATA = 30000

leX1 = preprocessing.LabelEncoder()
leX3 = preprocessing.LabelEncoder()
leX5 = preprocessing.LabelEncoder()
leX6 = preprocessing.LabelEncoder()
leX7 = preprocessing.LabelEncoder()
leX8 = preprocessing.LabelEncoder()
leX9 = preprocessing.LabelEncoder()
leX13 = preprocessing.LabelEncoder()
leY = preprocessing.LabelEncoder()
encX1 = preprocessing.OneHotEncoder(dtype = int)
encX3 = preprocessing.OneHotEncoder(dtype = int)
encX5 = preprocessing.OneHotEncoder(dtype = int)
encX6 = preprocessing.OneHotEncoder(dtype = int)
encX7 = preprocessing.OneHotEncoder(dtype = int)
encX8 = preprocessing.OneHotEncoder(dtype = int)
encX9 = preprocessing.OneHotEncoder(dtype = int)
encX13 = preprocessing.OneHotEncoder(dtype = int)
leX1.fit([x.strip() for x in "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked".split(',')])
leX3.fit([x.strip() for x in "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool".split(',')])
leX5.fit([x.strip() for x in "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse".split(',')])
leX6.fit([x.strip() for x in "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces".split(',')])
leX7.fit([x.strip() for x in "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried".split(',')])
leX8.fit([x.strip() for x in "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black".split(',')])
leX9.fit([x.strip() for x in "Female, Male".split(',')])
leX13.fit([x.strip() for x in "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split(',')])
leY.fit(["<=50K", ">50K"])

# X: features
# X0: age (continuous)
# X1: workclass
# X2: fnlwgt (continuous)
# X3: education
# X4: education-num (continuous)
# X5: marital-status
# X6: occupation
# X7: relationship
# X8: race
# X9: sex
# X10: capital-gain (continuous)
# X11: capital-loss (continuous)
# X12: hours-per-week (continuous)
# X13: native-country
data = [x.split(',') for x in open('adult.data').read().splitlines() if '?' not in x and x != '']
X0 = [[int(x[0])] for x in data]

# step 1. encode category labels
# step 2. one hot encoding category labels
X1 = leX1.transform([x[1].strip() for x in data])
X1 = encX1.fit_transform([[x] for x in X1]).toarray()

X2 = [[int(x[2])] for x in data]

X3 = leX3.transform([x[3].strip() for x in data])
X3 = encX3.fit_transform([[x] for x in X3]).toarray()

X4 = [[int(x[4])] for x in data]

X5 = leX5.transform([x[5].strip() for x in data])
X5 = encX5.fit_transform([[x] for x in X5]).toarray()

X6 = leX6.transform([x[6].strip() for x in data])
X6 = encX6.fit_transform([[x] for x in X6]).toarray()

X7 = leX7.transform([x[7].strip() for x in data])
X7 = encX7.fit_transform([[x] for x in X7]).toarray()

X8 = leX8.transform([x[8].strip() for x in data])
X8 = encX8.fit_transform([[x] for x in X8]).toarray()

X9 = leX9.transform([x[9].strip() for x in data])
X9 = encX9.fit_transform([[x] for x in X9]).toarray()

X10 = [[int(x[10])] for x in data]
X11 = [[int(x[11])] for x in data]
X12 = [[int(x[12])] for x in data]

X13 = leX13.transform([x[13].strip() for x in data])
X13 = encX13.fit_transform([[x] for x in X13]).toarray()
# concatenate categories
X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13), axis=1)
# Y: labels
Y = leY.transform([x[-1].strip() for x in data])

xTrain, xTest, yTrain, yTest = cv.train_test_split(X, Y, train_size = 5000/float(len(X)))


# In[3]:

import Classifiers as clfs
clfs.KNN(xTrain, xTest, yTrain, yTest)
clfs.RandomForest(xTrain, xTest, yTrain, yTest)
clfs.BoostedDecisionTree(xTrain, xTest, yTrain, yTest)
clfs.NeuralNets(xTrain, xTest, yTrain, yTest)
#clfs.SVM(xTrain, xTest, yTrain, yTest)
clfs.linearSVC(xTrain, xTest, yTrain, yTest)
clfs.XGBoost(xTrain, xTest, yTrain, yTest)


# In[ ]:




# In[ ]:



