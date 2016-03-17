CPU_CORES = 6
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from GenerateReport import generateReport
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

def KNN(xTrain, xTest, yTrain, yTest):
    print("---KNN report---")
    clf = KNeighborsClassifier()
    param_grid = {'n_neighbors' : np.arange(1,30) }
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=CPU_CORES)
    CV.fit(xTrain, yTrain)
    #print("best number of neighbors:" + str(CV.best_params_['n_neighbors']))
    generateReport(CV, xTest, yTest)

def RandomForest(xTrain, xTest, yTrain, yTest):
    print("---Random Forest report---")
    clf = RandomForestClassifier(n_estimators = 100)
    param_grid = {
        'max_features': [1,2,4,6,8,12,16]
    }
    CV = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=CPU_CORES)
    CV.fit(xTrain, yTrain)
    generateReport(CV, xTest, yTest)
    
def SVM(xTrain, xTest, yTrain, yTest):
    print("---SVM report---")

    svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    param_grid = {
        'C': [0.1, 1, 2, 3, 4, 5, 6, 7]
    }

    CV = GridSearchCV(svc, param_grid=param_grid, cv=2, n_jobs = CPU_CORES)
    CV.fit(xTrain, yTrain)

    generateReport(CV, xTest, yTest)

def BoostedDecisionTree(xTrain, xTest, yTrain, yTest):
    print("---Boosted Decision Tree report---")
    param_grid = {
        'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]#, 4096, 8192]
    }

    # classify
    clf = AdaBoostClassifier()
    CV = GridSearchCV(estimator=clf, param_grid = param_grid, cv = 5, n_jobs = CPU_CORES)
    CV.fit(X_train, Y_train)

    generateReport(CV, xTest, yTest)
    
def NeuralNets(xTrain, xTest, yTrain, yTest):
    print("---Neural Nets report---")
    param_grid = {
        'learning_rate_init': [10e-4, 10e-3, 10e-2]
    }
    clf = MLPClassifier(hidden_layer_sizes = (640,), algorithm='sgd', early_stopping = True, nesterovs_momentum = False, learning_rate = 'constant')
    CV = GridSearchCV(estimator=clf, param_grid = param_grid, cv = 5)
    CV.fit(X_train, Y_train)
    print("best learning_rate_init " + "{0:.3f}".format(CV.best_params_['learning_rate_init']))
    generateReport(CV, xTest, yTest)