# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:41:12 2025

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

#Model definition--------------------------------------------------------
nn = Sequential()

nn.add(Dense(1600, input_dim=833, activation='leaky_relu', kernel_initializer='normal'))
                
layer_sizes = [1600, 1200, 1000, 800, 400, 200,100]
dropout_rates = [0.25, 0.25, 0.2, 0.2, 0.15, 0.1, 0.05]

for size, drop in zip(layer_sizes, dropout_rates):
    nn.add(Dense(size,
                          activation='leaky_relu',
                          kernel_initializer='normal',
                          kernel_regularizer=regularizers.l2(0.00001)))
    nn.add(BatchNormalization()) 
    nn.add(Dropout(drop))        

nn.add(Dense(1,kernel_initializer='normal',activation="relu"))

learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate = 0.01,
    decay_steps = 500,
    decay_rate = 0.1,
    staircase=False,
    name="InverseTimeDecay",
)


opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)


nn.compile(loss="mean_squared_error",optimizer = opt,metrics=['mean_squared_error'])

nn.summary()


print("Fitting Model")

callback = tf.keras.callbacks.EarlyStopping(patience = 5)

history = nn.fit(np.array(X_train),
                          np.array(Y_train),
                          validation_split = 0.2,  
                          callbacks = [callback],
                          epochs=50)

print("Started evaluation MLP Regressor")
vd.evaluate_model(nn,X_train,X_test,Y_train,Y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])