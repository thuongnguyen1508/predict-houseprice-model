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

#######################                         #######################
#                       Creating the Model:                           #
#######################                         #######################

def get_model():
    model = models.Sequential([
        layers.Dense(10, input_shape = (6,), activation = 'relu'), #6 neurons, Input Layer
        layers.Dense(20, activation = 'relu'),                     #20 neurons, Hidden Layer
        layers.Dense(5, activation = 'relu'),                      #5  neurons, Hidden Layer
        layers.Dense(1)                                            #Output Layer
    ])                                                      #'relu' activation

    model.compile(
        loss='mse',                                         #Trained using Mean square error loss (Cost function) 
        optimizer='adam'                                    #Optimizer used is 'adam' (One of the Fastest optimizers)
    )
    
    return model

model = get_model()
model.summary()

#######################                         #######################
#                       Model Training:                               #
#######################                         #######################
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience = 5) #Defining early stopping parameter (optional, to save time)

model = get_model()

preds_on_untrained = model.predict(X_test) #Make predictions on the test set before training the parameters

#Finally training the model-->
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 100,
    callbacks = [early_stopping]
)


#######################                         #######################
#                       Save model                              #
#######################    
# 1. Save the trained model to a file
joblib.dump(model, 'neuronnetwork_price_predict_model.joblib')

# ... Later in your code or another script ...

# 2. Load the saved model
# loaded_model = joblib.load('trained_model.joblib')