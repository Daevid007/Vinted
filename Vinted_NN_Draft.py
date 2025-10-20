# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 18:30:12 2025

@author: david
"""

import numpy as np
import pandas as pd
import Vinted_Definitions as vd
import sklearn 

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


#------------------------------------------------------Adjustable-----------------------------------------------------

#File
name = "test"

#Pathes
git_path = "https://github.com/Daevid007/Vinted/blob/main/Data/"+name+"_data_parquet?raw=true"
local_path_1 = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"
#r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"

#-------------------------------------------------Data Cleaning/Preparation--------------------------------------------------

data = vd.load_data_parquet(name,git_path)

data_encoded = vd.encode_cols(data) 


#Defining Data
targets = data_encoded["Price"]
features = data_encoded.drop(columns="Price")
features = features.drop(columns = ["Title","ID","Link","Fees","Photos","Search_parameters"])


scaler = sklearn.preprocessing.StandardScaler()
targets_scaled = scaler.fit_transform(targets.to_frame())
targets_scaled = targets_scaled.flatten()

features_scaled = features
features_scaled[["Favourites","Dates","Time_Online_H","Favourites_per_hour"]] = scaler.fit_transform(features[["Favourites","Dates","Time_Online_H","Favourites_per_hour"]])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features_scaled,targets, test_size = 0.2)

print("Successfully prepared data")

#--------------------------------------------------------------------------------------------------------------------

mlp = MLPRegressor(
    hidden_layer_sizes=(236,118, 64, 32, 16),   # two hidden layers with 64 neurons each
    activation='relu',             # common choices: 'relu', 'tanh'
    solver='adam',                 # optimizer: 'adam' (default), 'lbfgs', or 'sgd'
    learning_rate='adaptive',
    max_iter=1000,                 # increase if training doesn’t converge
    random_state=42,
    early_stopping = True
)

print("Started Training")

mlp.fit(X_train, Y_train)

print("Started evaluation MLP Regressor")
vd.evaluate_model(mlp,X_train,X_test,Y_train,Y_test)

print("R2:")
print(mlp.score(X_test,Y_test))