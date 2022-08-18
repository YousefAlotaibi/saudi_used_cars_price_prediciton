import pandas as pd 
import numpy as np
import pickle

# Reading data
cars = pd.read_csv('/mnt/c/Users/youse/Desktop/miskdoc/misk-DSI/saudi_used_cars_price_prediciton/cars_train_test.csv')


# Splitting data to train and test
from sklearn.model_selection import train_test_split

X = cars.drop(['price'], axis=1)
Y = cars['price']


X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.3, random_state=123)

# Model building 

from xgboost import XGBRegressor
xgb = XGBRegressor(learning_rate=0.01, max_depth=6, n_estimators=1000,
             subsample=0.6)

# Training the model
xg = xgb.fit(X_train,Y_train)

# Prediction "testing"
#y_pred_xgb = xgb.predict(X_test)
# Model Aucraccy using R2
# r_squared = metrics.r2_score(Y_test,y_pred_xgb)
# print("R_squared :",r_squared)

with open('model_pkl', 'wb') as files:
    pickle.dump(xg, files)
#pickle.dump(xg, open('cars_pkl' , 'wb'))
