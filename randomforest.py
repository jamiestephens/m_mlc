# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:54:02 2021

@author: Jamie Stephens
"""
from numpy import mean
from numpy import std
import pandas as pd
import preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn
from matplotlib import pyplot as plt

def randomforest(iterationtype,weight):
    df = preprocessing.idf
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    maxd = 10

    # Hyperparameter tuning here:
    if weight == "":
        model = RandomForestClassifier(max_depth=maxd)
    else:
        model = RandomForestClassifier(class_weight=weight,max_depth=maxd)
        
    model.fit(X,y)
    y_pred_test = model.predict(X_test)
  #  print(accuracy_score(y_test, y_pred_test))
    
    cv = StratifiedKFold(n_splits=5)
    # evaluate the model on the dataset
    n_scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1,verbose=1)
    
    # report performance
    print(iterationtype)
    print('Mean ROC (AUC): %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
    conf_mat = confusion_matrix(y_test, y_pred_test)
    print(conf_mat)

    skplt.metrics.plot_confusion_matrix(y_test, y_pred_test)
    
    filename = iterationtype + '.png'
    print(filename)
    
    
    plt.savefig(filename, transparent=True) 
    plt.show()

if __name__ == "__main__":
    randomforest("Nobalancing","")
    randomforest("balanced","balanced")
    randomforest("1 to 10 balancing",{0:1,1:10})
    randomforest("1 to 20 balancing",{0:1,1:20})
    randomforest("1 to 40 balancing",{0:1,1:40})