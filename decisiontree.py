import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load your dataset
# Assume 'features' is a DataFrame containing input features, and 'labels' is a Series containing house prices
# Make sure to preprocess your data accordingly (e.g., handle missing values, encode categorical variables, etc.)
# For simplicity, let's assume the data is already preprocessed.
df = pd.read_csv('E:\School-Document\Datawarehouse\Assignment\Data_Set.csv')

X = df.iloc[:, :6] #Storing the features in 'X'
X.head()

Y = df.iloc[:, -1] #Storing the labels in 'Y'
Y.head()

X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape) #'shape' gives the dimension of the entity
print('Y_arr shape: ', Y_arr.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=0.2, random_state=42)

# Decision Tree
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

joblib.dump(decision_tree_model, 'decisiontree_price_predict_model.joblib')

# # Predictions on the test set
# y_pred_dt = decision_tree_model.predict(X_test)

# print(y_pred_dt)
# # Evaluate the Decision Tree model
# mse_dt = mean_squared_error(y_test, y_pred_dt)
# print(f"Decision Tree Mean Squared Error: {mse_dt}")
