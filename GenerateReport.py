import time
def generateReport(classifier, _xTest, _yTest):
    print ("Model accuracy: " + str(classifier.score(_xTest, _yTest)))
    
    t = time.time()
    y_pred = classifier.predict(_xTest)
    elapsed_time = time.time() - t
    
    print("prediction time: " + str(elapsed_time) + 's')
    
    from sklearn.metrics import f1_score
    f1_sco = f1_score(_yTest, y_pred)
    print("f1 score accuracy: " + str(f1_sco))

    from sklearn.metrics import classification_report
    print(classification_report(_yTest, y_pred, digits=4) )

    from sklearn.metrics import roc_auc_score
    print("area under roc score: " + str(roc_auc_score(_yTest, y_pred)))
    print("\n\n")