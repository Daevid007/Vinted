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


#Here I can vary
RF = sklearn.ensemble.RandomForestRegressor(n_estimators=200,max_depth = 15,oob_score=True)

targets = data_encoded["Price"]
features = data_encoded.drop(columns="Price")

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features,targets, test_size = 0.33)

X_train = X_train.drop(columns = ["Title","ID","Link","Fees","Photos","Search_parameters"])
X_test = X_test.drop(columns = ["Title","ID","Link","Fees","Photos","Search_parameters"])



RF.fit(X_train,Y_train)



#Benchmark

y_pred = np.full_like(Y_train, np.round(Y_train.mean(), 2), dtype=float)

print("Benchmark MSE:",sklearn.metrics.mean_squared_error(y_pred,Y_train))
print("Benchmark RMSE:",sklearn.metrics.mean_squared_error(y_pred,Y_train)**(1/2))
print("Benchmark MAE:",sklearn.metrics.mean_absolute_error(y_pred,Y_train))
print("MAPE:",sklearn.metrics.mean_absolute_percentage_error(y_pred,Y_train))


#Here we evaluate on the training set
print("Training Set",60*"-")
print("Obtained oob_score:" ,RF.oob_score_)


# Get feature importances
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


for x,y in zip(np.round(RF.predict(X_train)[0:10],2),Y_train[0:10]):
    print("Prediction on Train Set:",x,"Actual Y Target:",y)
    
print("MSE:",sklearn.metrics.mean_squared_error(np.round(RF.predict(X_train),2),Y_train))
print("RMSE:",sklearn.metrics.mean_squared_error(np.round(RF.predict(X_train),2),Y_train)**(1/2))
print("MAE:",sklearn.metrics.mean_absolute_error(np.round(RF.predict(X_train),2),Y_train))
print("MAPE:",sklearn.metrics.mean_absolute_percentage_error(np.round(RF.predict(X_train),2),Y_train))


#Here we evaluate on the test set

print("Test Set",60*"-")

for x,y in zip(np.round(RF.predict(X_test)[0:10],2),Y_test[0:10]):
    print("Prediction on Test Set:",x,"Actual Y Target:",y)
    
print("MSE:",sklearn.metrics.mean_squared_error(np.round(RF.predict(X_test),2),Y_test))
print("RMSE:",sklearn.metrics.mean_squared_error(np.round(RF.predict(X_test),2),Y_test)**(1/2))
print("MAE:",sklearn.metrics.mean_absolute_error(np.round(RF.predict(X_test),2),Y_test))
print("MAPE:",sklearn.metrics.mean_absolute_percentage_error(np.round(RF.predict(X_test),2),Y_test))





