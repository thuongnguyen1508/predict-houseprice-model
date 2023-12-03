import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import joblib
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks 
import matplotlib.pyplot as plt
import numpy as np

from chart import drawChart
df = pd.read_csv('E:\School-Document\Datawarehouse\Assignment\Data_Set.csv')

#######################                         #######################
#                      Data Normalization                             #
#######################                         #######################

df = df.iloc[:,1:] 
df_norm = (df - df.mean()) / df.std() #Data is normalized by subtracting each value in the column with the mean value and then dividing it with the standard                                                 deviation of the whole column
df_norm.head()
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):       #Defining a function which will convert the label values back to the original distribution and return it
    return int(pred * y_std + y_mean)

#######################                         #######################
#                      Creating Training and Test Sets                #
#######################                         #######################
X = df.iloc[:, :6] #Storing the features in 'X'
X.head()

Y = df.iloc[:, -1]
Y.head()

X_arr = X.values
Y_arr = Y.values


X_norm = df_norm.iloc[:, :6] #Storing the features in 'X'
X_norm.head()

Y_norm = df_norm.iloc[:, -1] #Storing the labels in 'Y'
Y_norm.head()

X_norm_arr = X_norm.values
Y_norm_arr = Y_norm.values

print('X_arr shape: ', X_norm_arr.shape) #'shape' gives the dimension of the entity
print('Y_arr shape: ', Y_norm_arr.shape)

#######################                         #######################
#                      Train and Test Split:                          #
#######################                         #######################

X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm_arr, Y_norm_arr, test_size = 0.05, shuffle = True, random_state=0) 
X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.05, shuffle = True, random_state=0) 
#This predefined function splits the dataset to train and test set, where test size is given in 'test_size'(Here 5%) 
#Random state ensures that the splits that you generate are reproducible. Scikit-learn uses random permutations to generate the splits.

# print('X_train shape: ', X_norm_train.shape)
# print('y_train shape: ', y_norm_train.shape)
# print('X_test shape: ', X_norm_test.shape)
# print('y_test shape: ', y_norm_test.shape)


# 2. Load the saved model
neuronnetwork_loaded_model = joblib.load('neuronnetwork_price_predict_model.joblib')
decisiontree_loaded_model = joblib.load('decisiontree_price_predict_model.joblib')
randomforest_loaded_model = joblib.load('randomforestregression_price_predict_model.joblib')


#######################                         #######################
#                       Final Prediction                              #
#######################                         #######################
neronnetwork_predict = neuronnetwork_loaded_model.predict(X_norm_test)
print(neronnetwork_predict)

decisiontree_predict = decisiontree_loaded_model.predict(X_test)
print(decisiontree_predict)

randomforest_predict = randomforest_loaded_model.predict(X_test)
print(randomforest_predict)

#######################                         #######################
#                      Plot Price Predictions::                       #
#######################                         #######################

# price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
print("DISPLAY CHART")
price_on_neuronnetworktrained = [convert_label_value(y[0]) for y in neronnetwork_predict]

drawChart(y_train, price_on_neuronnetworktrained, randomforest_predict, 50)
