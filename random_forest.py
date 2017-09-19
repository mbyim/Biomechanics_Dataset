import matplotlib
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

##RANDOM FOREST CLASSIFIER FOR MULTICLASS DATA##
#Code is also in a Kaggle Kernal


def split_data_pandas(file_name, train_sz = 0.1):
    data = pd.read_csv(file_name)
    data = shuffle(data)

    #Tested out some feature engineering based on some literature I saw, didn't perform well
    #data['ss_proportion'] = (data['pelvic_incidence'] - data['pelvic_tilt'])/data['pelvic_incidence']
    #data['pt_proportion'] = (1 - data['ss_proportion'])
  
    Y = data['class'].replace('Normal', 0).replace('Spondylolisthesis', 1).replace('Hernia', 2)
    #X = data.drop(['class', 'pelvic_tilt', 'sacral_slope'], axis = 1)
    X = data.drop('class', axis = 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = train_sz)
    X_test, X_dev, Y_test, Y_dev = train_test_split(X_test, Y_test, test_size = .5)
    
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

X_train, X_dev, X_test, Y_train, Y_dev, Y_test = split_data_pandas('../input/column_3C_weka.csv')

#Creating the model
model = RandomForestClassifier(n_estimators = 150, max_depth=5)
#Fit Model on training data
model.fit(X_train, Y_train)
print(X_train.columns)
print(model.feature_importances_)

#Predict on training
model.predict(X_train)
print(model.score(X_train, Y_train))

#Use model on Dev set
model.predict(X_dev)
print(model.score(X_dev, Y_dev))
