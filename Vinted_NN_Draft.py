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

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

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

cnn = Sequential()

print("Started Training")

cnn.fit(X_train, Y_train)

print("Started evaluation MLP Regressor")
vd.evaluate_model(cnn,X_train,X_test,Y_train,Y_test)

print("R2:")
