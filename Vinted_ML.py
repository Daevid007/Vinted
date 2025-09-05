# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:05:23 2025

@author: david
"""
import numpy as np
import pandas as pd
import Vinted_Definitions as vd
import sklearn 

data_raw = vd.load_data_parquet("test")
data_encoded = vd.encode_cols(data_raw) #Probably not enough data yet to use all columns...

#Defining models
RF = sklearn.ensemble.RandomForestRegressor(n_estimators=200,max_depth = 15,oob_score=True)


#Defining Data
targets = data_encoded["Price"]
features = data_encoded.drop(columns="Price")
features = features.drop(columns = ["Title","ID","Link","Fees","Photos","Search_parameters"])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features,targets, test_size = 0.33)

scaler = sklearn.preprocessing.StandardScaler()
targets_scaled = scaler.fit_transform(targets.to_frame())
targets_scaled = targets_scaled.flatten()

features_scaled = features
features_scaled[["Favourites","Dates","Time_Online_H","Favourites_per_hour"]] = scaler.fit_transform(features[["Favourites","Dates","Time_Online_H","Favourites_per_hour"]])


#Training models

RF.fit(X_train,Y_train)

#Evaluation

vd.evaluate_model(RF,X_train,X_test,Y_train,Y_test)

# Get feature importances
print("Feature Importances",60*"-")
importances = RF.feature_importances_
feature_names = X_train.columns

# Put into a DataFrame
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Sort and get top 10
top10 = feat_imp.sort_values(by="importance", ascending=False).head(10)

print("Top 10 features:",top10)


