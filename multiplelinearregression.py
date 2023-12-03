import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix


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

X = df_norm.iloc[:, :6] #Storing the features in 'X'
X.head()

Y = df_norm.iloc[:, -1] #Storing the labels in 'Y'
Y.head()

X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape) #'shape' gives the dimension of the entity
print('Y_arr shape: ', Y_arr.shape)

#######################                         #######################
#                      Train and Test Split:                          #
#######################                         #######################

X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.05, shuffle = True, random_state=0) 
#This predefined function splits the dataset to train and test set, where test size is given in 'test_size'(Here 5%) 
#Random state ensures that the splits that you generate are reproducible. Scikit-learn uses random permutations to generate the splits.

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


mlr = LinearRegression()
mlr.fit(X_train,y_train)
mlr_score = mlr.score(X_test,y_test)
pred_mlr = mlr.predict(X_test)


print("Result-----------------------------------")
print(pred_mlr)
# expl_mlr = explained_variance_score(pred_mlr,y_test)