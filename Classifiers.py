CPU_CORES = 32
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from GenerateReport import generateReport
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
import time
    
def KNN(xTrain, xTest, yTrain, yTest):
    print("---KNN report---")
    clf = KNeighborsClassifier()
    param_grid = {'n_neighbors' : [x for x in np.arange(1,30) if x%2 == 1] }
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=CPU_CORES)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best number of neighbors:" + str(CV.best_params_['n_neighbors']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)

def RandomForest(xTrain, xTest, yTrain, yTest):
    print("---Random Forest report---")
    clf = RandomForestClassifier(n_estimators = 100)
    param_grid = {
        'max_features': [x for x in [1,2,4,6,8,12,16,20] if x < len(xTrain[0])]
    }
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=CPU_CORES)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best number of max_features:" + str(CV.best_params_['max_features']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)
    
def SVM(xTrain, xTest, yTrain, yTest):
    print("---SVM with Linear kernel (LibSVM) report---")
    svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    param_grid = {
        'C': [0.1, 1, 2, 3, 4, 5, 6, 7]
    }

    CV = GridSearchCV(svc, param_grid=param_grid, cv=5, n_jobs = CPU_CORES)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best number of C:" + str(CV.best_params_['C']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)
    
def linearSVC(xTrain, xTest, yTrain, yTest):
    print("---LinearSVM report---")

    svc = LinearSVC(C=1.0, class_weight=None,
        max_iter=-1, random_state=None,
        tol=0.001, verbose=False)

    param_grid = {
        'C': [0.1, 1, 2, 3, 4, 5, 6, 7]
    }

    CV = GridSearchCV(svc, param_grid=param_grid, cv=5, n_jobs = CPU_CORES)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best number of C:" + str(CV.best_params_['C']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)
    
def BoostedDecisionTree(xTrain, xTest, yTrain, yTest):
    print("---Boosted Decision Tree report---")
    param_grid = {
        'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    }

    # classify
    clf = AdaBoostClassifier()
    CV = GridSearchCV(estimator=clf, param_grid = param_grid, cv = 5, n_jobs = CPU_CORES)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best number of n_estimators:" + str(CV.best_params_['n_estimators']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)

def XGBoost(xTrain, xTest, yTrain, yTest):
    xTrain = np.array(xTrain);
    xTest = np.array(xTest);
    yTrain = np.array(yTrain);
    yTest = np.array(yTest);
    
    print("---XGBoost report---")
    param_grid = {
        'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    }
        
    clf = xgb.XGBClassifier()
    CV = GridSearchCV(estimator=clf, param_grid = param_grid, cv = 5)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best number of n_estimators:" + str(CV.best_params_['n_estimators']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)
    
    
def NeuralNets(xTrain, xTest, yTrain, yTest):
    print("---Neural Nets report---")
    param_grid = {
        'learning_rate_init': [10e-4, 10e-3, 10e-2]
    }
    clf = MLPClassifier(hidden_layer_sizes = (640,), algorithm='sgd', early_stopping = True, nesterovs_momentum = False, learning_rate = 'constant')
    CV = GridSearchCV(estimator=clf, param_grid = param_grid, cv = 5)
    t = time.time()
    CV.fit(xTrain, yTrain)
    elapsed_time = time.time() - t
    print("best learning_rate_init " + "{0:.3f}".format(CV.best_params_['learning_rate_init']))
    print("training time: " + str(elapsed_time) + 's')
    generateReport(CV, xTest, yTest)