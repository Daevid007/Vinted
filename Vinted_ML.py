# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:05:23 2025

@author: david
"""
import numpy as np
import pandas as pd
import Vinted_Definitions as vd
import sklearn 

#------------------------------------------------------Adjustable-----------------------------------------------------

#File
name = "test"

#Pathes
git_path = "https://github.com/Daevid007/Vinted/blob/main/Data/"+name+"_data_parquet?raw=true"
local_path_1 = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"
#r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"

#--------------------------------------------------------------------------------------------------------------------

data = vd.load_data_parquet(name,git_path)

data_encoded = vd.encode_cols(data) 


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


#Defining models
RF = sklearn.ensemble.RandomForestRegressor(n_estimators=200, max_depth = 15 ,min_samples_leaf=2, oob_score=True)

params_reg = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 4,
    
    "learning_rate": 0.03,
    "loss": "squared_error",
}

reg = sklearn.ensemble.GradientBoostingRegressor(**params_reg)

Lreg = sklearn.linear_model.LinearRegression(fit_intercept = False)

#Training models
Lreg.fit(X_train,Y_train)
RF.fit(X_train,Y_train)
reg.fit(X_train,Y_train)
#Evaluation


print("Random Forest Regressor",60*"-")

vd.evaluate_model(RF,X_train,X_test,Y_train,Y_test)

print("Random Forest Regressor OOB Score",60*"-")

if RF.oob_score == True:
    print("Obtained oob_score:" ,RF.oob_score_)
else:
    print("No oob score available")

# Get feature importances
print("Feature Importances",60*"-")
print("Random Forest Regressor",60*"-")
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


print("Boosted Regression Model",60*"-")

vd.evaluate_model(reg,X_train,X_test,Y_train,Y_test)
# Get feature importances
print("Feature Importances",60*"-")
print("Boosted Regression Model",60*"-")
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


print("Linear Regressor",60*"-")
vd.evaluate_model(Lreg,X_train,X_test,Y_train,Y_test)
