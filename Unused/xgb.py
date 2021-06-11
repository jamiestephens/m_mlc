# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:58:44 2021

@author: Administrator
"""

import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
import xgboost as xgb
import featureselection
from sklearn.model_selection import train_test_split

df = featureselection.df
data = df.values
X, y = data[:, :-1], data[:, -1]
print("got here okay")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2)

gbm = xgb.XGBClassifier( 
                        n_estimators=30000,
                        max_depth=4,
                        objective='binary:logistic', #new objective
                        learning_rate=.05, 
                        subsample=.8,
                        min_child_weight=3,
                        colsample_bytree=.8
                       )

eval_set=[(X_train,y_train),(X_val,y_val)]
fit_model = gbm.fit( 
                    X_train, y_train, 
                    eval_set=eval_set,
                    eval_metric='error', #new evaluation metric: classification error (could also use AUC, e.g.)
                    early_stopping_rounds=50,
                    verbose=False
                   )

print(accuracy_score(y_test, gbm.predict(X_test, ntree_limit=gbm.best_ntree_limit)))

