# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:12:56 2025

@author: david
"""

import os
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from pyVinted import Vinted
import pandas as pd
import numpy as np
import sklearn
import datetime
import json
import time
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import datetime
from datetime import timedelta

#Setting the API up
vinted = Vinted()
testitems = vinted.items.search("https://www.vinted.de/catalog?catalog[]=79&catalog_from=0&brand_ids[]=53&page=1&order=relevance",10,1)
testitem1 = testitems[0]

def SearchVintedDraft(order = "relevance",price_to = "60",currency = "EUR",text = "Adidas-Vintage", page = 1, catalog = "77"):
    """
    Function to search on Vinted after specified search criteria, returning a list of items
    Max number of items before Client error is around 950
    """

    string = "https://www.vinted.de/catalog?order="+order+"&price_to="+price_to+"&currency="+currency+"&search_text="+text+"&catalog[]="+catalog
    return vinted.items.search(string,900,page)



def showproperties():
    """
    Function to show the raw data of one item searched, returning a dictionary
    Purpose only is looking stuff up
    """
    return vinted.items.search("https://www.vinted.de/vetement?order=newest_first&price_to=60&currency=EUR&search_text=Adidas-Vintage",1,1)[0].raw_data

def show_pic(url,title):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Display image
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

"""# Here define functions to work with the data parquet (which has to be down and uploaded manually in the notebook environment)"""

def save_data_parquet(name, data):
    """
    Creates a new data parquet to store the data of found items
    """
    parquet_file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet" #HAS TO BE CHANGED EVENTUALLY DEPENDING ON DEVICE
    data.to_parquet(parquet_file_path, index=False, compression='snappy')
    print(f"\nDataFrame successfully saved to {parquet_file_path}")



def create_data_parquet(name):
    data = pd.DataFrame({"ID":[],
                        "Title": [],
                        "Price":[],
                        "Favourites":[],
                        "Link":[],
                        "Brand Title":[],
                        "Promoted":[],
                        "Status":[],
                        "Fees":[],
                        "Dates":[],
                        "Photos":[],
                        "Size":[],
                        "Search_parameters":[]})
    
    
    parquet_file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet"   #HAS TO BE CHANGED EVENTUALLY DEPENDING ON DEVICE
    data = data.drop_duplicates(subset=['ID'])
    data.to_parquet(parquet_file_path, index=False, compression='snappy')
    print(f"\nDataFrame successfully saved to {parquet_file_path}")
    
    if os.path.exists(parquet_file_path):
        print(f"{parquet_file_path} created successfully.")
    else:
        print(f"{parquet_file_path} created not successfully.")

    
    return data
    
def delete_data_parquet(name):
    
    # Path to the file
    file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet"
    
    # Check if file exists before deleting
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted successfully.")
    else:
        print(f"{file_path} does not exist.")

    
def load_data_parquet(name):
    """
    Loads a specified data_parquet into a dataframe and returns it
    """
    return pd.read_parquet("C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet")  #HAS TO BE CHANGED EVENTUALLY DEPENDING ON DEVICE


def add_data_to_parquet(name,data):
    """
    Adds search data to an existing data parquet, saves this parquet and returns the merged DataFrame
    Also removes duplicates
    """
    tmp = pd.read_parquet("C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet")
    data = pd.concat([tmp,data])
    parquet_file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet"   #HAS TO BE CHANGED EVENTUALLY DEPENDING ON DEVICE
    data = data.drop_duplicates(subset=['ID'])
    data.to_parquet(parquet_file_path, index=False, compression='snappy')
    print(f"\nDataFrame successfully saved to {parquet_file_path}")
    return data


def create_search_df(items_searched,search_parameters):
  df = pd.DataFrame({"ID":[],
                    "Title": [],
                    "Price":[],
                    "Favourites":[],
                    "Link":[],
                    "Brand Title":[],
                    "Promoted":[],
                    "Status":[],
                    "Fees":[],
                    "Dates":[],
                    "Photos":[],
                    "Size":[],
                    "Search_parameters":[]})
  #------------------------------------------------------------------------------------------------------------------
    


  #storing the retrieved data in a df format

  titles = [item.title for item in items_searched]
  IDs = [item.id for item in items_searched]
  prices = [item.price for item in items_searched]
  favourites = [item.raw_data["favourite_count"] for item in items_searched]
  Links = [item.url for item in items_searched]
  Brands = [item.brand_title for item in items_searched]
  Promoted = [item.raw_data["promoted"] for item in items_searched]
  Status = [item.raw_data["status"] for item in items_searched]
  Fees = [item.raw_data["service_fee"]["amount"] for item in items_searched]
  Dates = [datetime.datetime.fromtimestamp(item.raw_timestamp) for item in items_searched]
  photos = [item.photo for item in items_searched]
  sizes = [item.size_title for item in items_searched]
  
  
  df["ID"] = IDs
  df["Title"] = titles
  df["Price"] = prices
  df["Favourites"] = favourites
  df["Link"] = Links
  df["Brand Title"] = Brands
  df["Promoted"] = Promoted
  df["Status"] = Status
  df["Fees"] = Fees
  df["Dates"] = Dates
  df["Photos"] = photos
  df["Size"] = sizes
  df["Search_parameters"] = [search_parameters]*len(df)

  return df




def collect_data(name = "test",order = "relevance",price_to = "60",currency = "EUR",parameters_text = ["Adidas-Vintage","Nike-Vintage"], pages = 1,catalog = "77"):
  start_parquet_size = load_data_parquet("test").shape[0] #Saving start data size
  print("Start parquet size:",start_parquet_size)
  counter = 0
  for i in (parameters_text): #We iterate trough the searched text parameters and...
    counter += 1
    for j in range(1,pages+1):
      print("Parameter:",i, "Page:",j, "from",pages, "Parameter", counter, "from", len(parameters_text)) #Counter to keep track of progress
      items_searched = SearchVintedDraft(order = order, price_to = price_to, currency = currency, text = i, page = j,catalog = catalog) #Simply search for the items
      search_parameters = {"order":order
                           ,"price_to":price_to
                           ,"currency":currency
                           ,"Page":j,
                           "catalog":catalog}
      df = create_search_df(items_searched,search_parameters) #Creating the df for the searched items
      #---------------------------------
      current_time = datetime.datetime.now()
      df["Time_Online"] = (current_time - df["Dates"])
      df['Time_Online_H'] = df['Time_Online'].dt.total_seconds() / 3600
      df = df.drop("Time_Online", axis=1)
      df["Favourites_per_hour"] = df["Favourites"]/df["Time_Online_H"]
      df.ID = df.ID.astype(int).astype(str)
      #---------------------------------
      add_data_to_parquet(name,df) # Adding the data to the parquet
      print(df.head()[["ID","Title"]])
      print()
      time.sleep(5) #To avoid errors...
      #--------------- Here I should add some prevention so the web scraping does not go well or so

  end_parquet_size = load_data_parquet("test").shape[0] #Saving end data size
  print("End parquet size:",end_parquet_size) #Print end size
  print("New data collected:",end_parquet_size-start_parquet_size) #Print how much was added




def check_clean_data(name ="test"):
    df = load_data_parquet(name)
    print("Shape before:",df.shape)
    df = df.drop_duplicates(subset=['ID'])
    print("Shape:",df.shape)
    return df


def encode_cols(df):
    # Create an instance of the encoder, drop='first' to avoid multicollinearity, sparse_output=False for dense array
    Encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Select the columns you want to use as features
    features_df = df
    features_df.Dates = features_df.Dates.astype(int) // 10**9
    # Use fit_transform on the instance to encode the 'Status' column
    encoded_status = Encoder.fit_transform(features_df[["Status","Size","Brand Title"]])
    
    # Get the names of the encoded categories (excluding the dropped one)
    encoded_category_names = Encoder.get_feature_names_out(["Status","Size","Brand Title"])
    
    # Create a DataFrame from the encoded status data with appropriate column names
    encoded_status_df = pd.DataFrame(encoded_status, columns=encoded_category_names, index=features_df.index)
    
    # Drop the original 'Status' column from the features DataFrame
    features_df = features_df.drop(["Status","Size","Brand Title"], axis=1)
    
    # Concatenate the original features (without 'Status') and the encoded status columns
    data_encoded = pd.concat([features_df, encoded_status_df], axis=1)

    
    data_encoded['Promoted'] = data_encoded['Promoted'].astype(int)    
    data_encoded['Price'] = pd.to_numeric(data_encoded['Price'], errors='coerce')
    data_encoded['Fees'] = pd.to_numeric(data_encoded['Fees'], errors='coerce')
    
    print(data_encoded.head())
    
    return data_encoded



def evaluate_model(model,X_train,X_test,Y_train,Y_test):
    """
    Parameters: model, X_train, X_test, Y_train, Y_test
    Returns: None
    """
    y_pred = np.full_like(Y_train, np.round(Y_train.mean(), 2), dtype=float)

    print("Benchmark MSE:",sklearn.metrics.mean_squared_error(y_pred,Y_train))
    print("Benchmark RMSE:",sklearn.metrics.mean_squared_error(y_pred,Y_train)**(1/2))
    print("Benchmark MAE:",sklearn.metrics.mean_absolute_error(y_pred,Y_train))
    print("MAPE:",sklearn.metrics.mean_absolute_percentage_error(y_pred,Y_train))


    #Here we evaluate on the training set
    print("Training Set",60*"-")
    if model.oob_score == True:
        print("Obtained oob_score:" ,model.oob_score_)
    else:
        print("No oob score available")


    for x,y in zip(np.round(model.predict(X_train)[0:10],2),Y_train[0:10]):
        print("Prediction on Train Set:",x,"Actual Y Target:",y)
        
    print("MSE:",sklearn.metrics.mean_squared_error(np.round(model.predict(X_train),2),Y_train))
    print("RMSE:",sklearn.metrics.mean_squared_error(np.round(model.predict(X_train),2),Y_train)**(1/2))
    print("MAE:",sklearn.metrics.mean_absolute_error(np.round(model.predict(X_train),2),Y_train))
    print("MAPE:",sklearn.metrics.mean_absolute_percentage_error(np.round(model.predict(X_train),2),Y_train))


    #Here we evaluate on the test set

    print("Test Set",60*"-")

    for x,y in zip(np.round(model.predict(X_test)[0:10],2),Y_test[0:10]):
        print("Prediction on Test Set:",x,"Actual Y Target:",y)
        
    print("MSE:",sklearn.metrics.mean_squared_error(np.round(model.predict(X_test),2),Y_test))
    print("RMSE:",sklearn.metrics.mean_squared_error(np.round(model.predict(X_test),2),Y_test)**(1/2))
    print("MAE:",sklearn.metrics.mean_absolute_error(np.round(model.predict(X_test),2),Y_test))
    print("MAPE:",sklearn.metrics.mean_absolute_percentage_error(np.round(model.predict(X_test),2),Y_test))


def heatmap_vinted(data_encoded, numerical_cols):

    df_for_heatmap = data_encoded[numerical_cols]

    # Calculate the correlation matrix
    correlation_matrix = df_for_heatmap.corr()

    # Create the heatmap
    plt.figure(figsize=(10, 8)) # Set the figure size for better readability
    sns.heatmap(
        correlation_matrix,
        annot=True,      # Show the correlation values on the heatmap
        cmap='plasma', # Choose a color map (e.g., 'viridis', 'plasma', 'coolwarm')
        fmt=".2f",       # Format the annotations to two decimal places
        linewidths=.5,   # Add lines between cells
        cbar=True        # Show the color bar
    )
    plt.title('Correlation Heatmap of Vinted Items Components') # Add a title to the heatmap
    plt.show() # Display the plot
    print("Plots generated successfully!")

def scatter_favourites(data_encoded,numerical_cols):
    # Create scatter plots for 'Favourites' against other numerical columns
    for col in numerical_cols:
        if col != 'Favourites': # Plot 'Favourites' against every other column
            plt.figure(figsize=(7, 5))
            sns.scatterplot(x=data_encoded[col], y=data_encoded['Favourites'], alpha=0.6)
            plt.title(f'Favourites vs. {col}')
            plt.xlabel(col)
            plt.xticks(rotation=45) # Rotate by 45 degrees
            plt.ylabel('Favourites Count')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

    for col in numerical_cols:
        if col != 'Favourites' and col != "Price" and col != "Fees": # Plot 'Favourites' against every other column
            plt.figure(figsize=(8, 6)) # Adjust figure size for better readability

            # Check if the column is more categorical/binary or continuous
            # A simple heuristic: if unique values are few, treat as categorical
            # Otherwise, treat as continuous and aggregate
            if data_encoded[col].nunique() < 10 or 'Status_' in col or col == 'Promoted':
                # Use pointplot for categorical/binary variables to show mean and CI
                sns.pointplot(x=data_encoded[col], y=data_encoded['Favourites'], linestyles=None, errorbar='sd', capsize=0.1)
                plt.title(f'Mean Favourites by {col}')
                plt.xlabel(col)
                plt.ylabel('Mean Favourites Count')
                plt.xticks(rotation=45, ha='right') # Rotate for categorical labels
            else:
                # For continuous variables, calculate mean Favourites for each unique X value
                # This aggregates the data before plotting
                mean_favourites_by_x = data_encoded.groupby(col)['Favourites'].mean().reset_index()
                sns.scatterplot(x=mean_favourites_by_x[col], y=mean_favourites_by_x['Favourites'], s=100, color='blue', alpha=0.8)
                plt.title(f'Mean Favourites vs. {col} (Aggregated)')
                plt.xlabel(col)
                plt.ylabel('Mean Favourites Count')
                plt.xticks(rotation=45) # Rotate for continuous labels

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout() # Adjust layout to prevent labels from running off
            plt.show()
    print("Plots generated successfully!")


def plots_price_fees(data_encoded,numerical_cols):    
    for col_rel in ['Price', 'Fees']:
        plt.figure(figsize=(8, 6))
        # Use regplot to show scatter points and a linear regression line
        sns.regplot(x=data_encoded[col_rel], y=data_encoded['Favourites'], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        plt.title(f'Relationship between {col_rel} and Favourites')
        plt.xlabel(col_rel)
        plt.ylabel('Favourites Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    
    """# I do the same with Price as the main focus point:"""
    
    # Convert 'Price' and 'Fees' columns to numeric
    data_encoded['Price'] = pd.to_numeric(data_encoded['Price'], errors='coerce')
    data_encoded['Fees'] = pd.to_numeric(data_encoded['Fees'], errors='coerce')
    
    # Create scatter plots for 'Price' against other numerical columns
    for col in numerical_cols:
        if col != 'Price': # Plot 'Price' against every other column
            plt.figure(figsize=(7, 5))
            sns.scatterplot(x=data_encoded[col], y=data_encoded['Price'], alpha=0.6)
            plt.title(f'Price vs. {col}')
            plt.xlabel(col)
            plt.xticks(rotation=45) # Rotate by 45 degrees
            plt.ylabel('Price Count')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
    
    
    for col in numerical_cols:
        if col != 'Price' and col != "Favourites" and col != "Fees": # Plot 'Price' against every other column
            plt.figure(figsize=(8, 6)) # Adjust figure size for better readability
    
            # Check if the column is more categorical/binary or continuous
            # A simple heuristic: if unique values are few, treat as categorical
            # Otherwise, treat as continuous and aggregate
            if data_encoded[col].nunique() < 10 or 'Status_' in col or col == 'Promoted':
                # Use pointplot for categorical/binary variables to show mean and CI
                sns.pointplot(x=data_encoded[col], y=data_encoded['Price'], linestyles=None, errorbar='sd', capsize=0.1)
                plt.title(f'Mean Price by {col}')
                plt.xlabel(col)
                plt.ylabel('Mean Price Count')
                plt.xticks(rotation=45, ha='right') # Rotate for categorical labels
            else:
                # For continuous variables, calculate mean Price for each unique X value
                # This aggregates the data before plotting
                mean_price_by_x = data_encoded.groupby(col)['Price'].mean().reset_index()
                sns.scatterplot(x=mean_price_by_x[col], y=mean_price_by_x['Price'], s=100, color='blue', alpha=0.8)
                plt.title(f'Mean Price vs. {col} (Aggregated)')
                plt.xlabel(col)
                plt.ylabel('Mean Price Count')
                plt.xticks(rotation=45) # Rotate for continuous labels
    
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout() # Adjust layout to prevent labels from running off
            plt.show()
    
    
    
    
    for col_rel in ['Favourites', 'Fees']:
        plt.figure(figsize=(8, 6))
        # Use regplot to show scatter points and a linear regression line
        sns.regplot(x=data_encoded[col_rel], y=data_encoded['Price'], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        plt.title(f'Relationship between {col_rel} and Price')
        plt.xlabel(col_rel)
        plt.ylabel('Price Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    print("Plots generated successfully!")