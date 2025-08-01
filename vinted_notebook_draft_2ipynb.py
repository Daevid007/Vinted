# -*- coding: utf-8 -*-
"""Vinted_notebook_draft.2ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1deYvqg0hryJZqPSdkbtotJOoP1zBJYtM

# Installing modules and importing Libraries
"""



import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from pyVinted import Vinted
import pandas as pd
import datetime
import json
import time
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import datetime
from datetime import timedelta

"""# Setting up the class to search for items and testing it"""

#Declaring vinted as Vinted class and testing items search
vinted = Vinted()
items = vinted.items.search("https://www.vinted.de/catalog?catalog[]=79&catalog_from=0&brand_ids[]=53&page=1&order=relevance",10,1)
item1 = items[0]

"""# Defining functions to use vinted items search function properly"""

#I can add parameters like search_text etc in the url string manually or with a function like the following to expand the capabilities.

def SearchVintedDraft(order = "relevance",price_to = "60",currency = "EUR",text = "Adidas-Vintage", page = 1):
    """
    Function to search on Vinted after specified search criteria, returning a list of items
    Max number of items before Client error is around 950
    """

    string = "https://www.vinted.de/catalog?order="+order+"&price_to="+price_to+"&currency="+currency+"&search_text="+text
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

"""# Here we search (default) for items, convert them into a df format and adding them to the data parquet"""

#------------------------------------------------------------------------------------------------------------------
#to work with the pictures
items_searched = SearchVintedDraft()

def create_search_df(items_searched):
#-----------------------------------------

#I should also add size here!

#------------------------------------------
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
                    "Photos":[]})
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

  return df

#I should also add the size parameter

#------------------------------------------------------------------------------------------------------------------


#Next step is to adding to the data via different criteria
#And to define the criteria structure in the json file
#Also finding out if views can be used somehow and finding out how to get the description text and not just the titles
#Checking how to use ids to find a sold item and verify if its sold
#------------------------------------------------------------------------------------------------------------------

#Adding all adittional data to the test_data_parquet

df_new_data = create_search_df(items_searched)
data = add_data_to_parquet("test", df_new_data)

"""# Now we will use a loop and add some data to the parquet."""

def collect_data(order = "relevance",price_to = "60",currency = "EUR",parameters_text = ["Adidas-Vintage","Nike-Vintage"], pages = 1):
  start_parquet_size = load_data_parquet("test").shape[0] #Saving start data size
  print("Start parquet size:",start_parquet_size)
  counter = 0
  for i in (parameters_text): #We iterate trough the searched text parameters and...
    counter += 1
    for j in range(1,pages+1):
      print("Parameter:",i, "Page:",j, "from",pages, "Parameter", counter, "from", len(parameters_text)) #Counter to keep track of progress
      items_searched = SearchVintedDraft(order = order, price_to = price_to, currency = currency, text = i, page = j) #Simply search for the items
      df = create_search_df(items_searched) #Creating the df for the searched items
      #---------------------------------
      current_time = datetime.datetime.now()
      df["Time_Online"] = (current_time - df["Dates"])
      df['Time_Online_H'] = df['Time_Online'].dt.total_seconds() / 3600
      df = df.drop("Time_Online", axis=1)
      df["Favourites_per_hour"] = df["Favourites"]/df["Time_Online_H"]
      #---------------------------------
      add_data_to_parquet("test",df) # Adding the data to the parquet
      print(df.head()[["ID","Title"]])
      print()
      time.sleep(5) #To avoid errors...
      #--------------- Here I should add some prevention so the web scraping does not go well or so

  end_parquet_size = load_data_parquet("test").shape[0] #Saving end data size
  print("End parquet size:",end_parquet_size) #Print end size
  print("New data collected:",end_parquet_size-start_parquet_size) #Print how much was added

collect_data(pages = 1, parameters_text = ["Adidas-Vintage Shirt","Nike-Vintage Shirt","Puma-Vintage Shirt","Vintage Shirts","Umbro-Vintage Shirt","Kappa-Vintage Shirt","Vintage T-Shirt","WRSTBHVR-Shirt","Reebook Shirt"])
#Default use, collecting new data to our data parquet, only 1 page seems to be possible... probably because there is just one

"""## Here we check the full data (which is not checked yet on if it can still be found or not

Either I try to get every single item by its ID which will take a long time and probably lead to an HTTP Error, or I just do the exact same search again and compare which old IDs are not in the new IDs assuming that with
one Search I get all the items which seems unlikely for e.g. Adidas-Vintage...
I could also use the parameters to filter for every single item and search them one by one and then compare the ID with the found IDs of the search, however this is not efficient and leads probably to a HTTTP Error too.




#First we want the oldest items
df = load_data_parquet("test")
df = df.sort_values(by="Dates")
print(df.head())


def SearchVintedDraft(order = "relevance",price_to = "60", price_from = "60",currency = "EUR",text = "Adidas-Vintage", page = 1,brand_title = "Adidas"):

    #Function to search on Vinted after specified search criteria, returning a list of items
    #Max number of items before Client error is around 950


    string = "https://www.vinted.de/catalog?order="+order+"&price_to="+price_to+"&currency="+currency+"&search_text="+text#+"&price_from="+price_from+"&brand="+brand_title
    return vinted.items.search(string,900,page)


items_searched = SearchVintedDraft()
df_verification = create_search_df(items_searched)
print(df_verification)
#https://www.vinted.de/catalog?search_text=Polo%20Vintage%20Kappa&time=1753278277&search_id=25081498161&page=2&currency=EUR&price_from=5&price_to=5&brand_ids[]=8139&status_ids[]=2

# Here we add colums an Online for X.Y Hours and a Favourites per Hour Count
Even though I do the same in the collect data definition this here can be used to update all the data, this is esspecially useful since the data from the data collect definition will be relatively new (freshly searched) and therefore biased
"""

#convert_duration_string_to_hours(duration) for full data

def convert_to_time_online(name = "test"):
    df_tmp = load_data_parquet(name)
    df_tmp["Dates"]
    
    current_time = datetime.datetime.now()
    
    df_tmp["Time_Online"] = (current_time - df_tmp["Dates"])
    df_tmp['Time_Online_H'] = df_tmp['Time_Online'].dt.total_seconds() / 3600
    df_tmp = df_tmp.drop("Time_Online", axis=1)
    df_tmp["Favourites_per_hour"] = df_tmp["Favourites"]/df_tmp["Time_Online_H"]
    print(df_tmp)
    
    save_data_parquet(name, df_tmp)
    
convert_to_time_online()

"""# From here we will use the data
So we load and show it and remove duplicates to be safe
"""
def check_clean_data(name ="test"):
    df = load_data_parquet(name)
    print("Shape before:",df.shape)
    df = df.drop_duplicates(subset=['ID'])
    print("Shape:",df.shape)
    return df

df = check_clean_data()

"""# Now I do some fundamental analysis like correlations key features and

First we encode the Status column:
"""

#------------------------------------To do change to functions-------------------------------------------------------------------------------------------

def encode_status(df = df):
    # Create an instance of the encoder, drop='first' to avoid multicollinearity, sparse_output=False for dense array
    Encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Select the columns you want to use as features
    features_df = df[["Title","Price", "Favourites", "Promoted", "Status", "Fees","Brand Title","Time_Online_H","Favourites_per_hour"]]
    
    # Use fit_transform on the instance to encode the 'Status' column
    encoded_status = Encoder.fit_transform(features_df[["Status"]])
    
    # Get the names of the encoded categories (excluding the dropped one)
    encoded_category_names = Encoder.get_feature_names_out(["Status"])
    
    # Create a DataFrame from the encoded status data with appropriate column names
    encoded_status_df = pd.DataFrame(encoded_status, columns=encoded_category_names, index=features_df.index)
    
    # Drop the original 'Status' column from the features DataFrame
    features_df = features_df.drop("Status", axis=1)
    
    # Concatenate the original features (without 'Status') and the encoded status columns
    data_encoded = pd.concat([features_df, encoded_status_df], axis=1)
    
    print(data_encoded.head())
    
    return data_encoded

data_encoded = encode_status()


"""Then the brand column"""

def encode_brands(df,encode = True):
    if encode == True:  
        # Create an instance of the encoder, drop='first' to avoid multicollinearity, sparse_output=False for dense array
        Encoder = OneHotEncoder(drop='first', sparse_output=False)
          
        # Select the columns you want to use as features
        features_df = df[["Title","Price", "Favourites", "Promoted", "Status_Neu","Status_Neu, mit Etikett","Status_Sehr gut","Status_Zufriedenstellend", "Fees","Brand Title","Favourites_per_hour","Time_Online_H"]]
          
        # Use fit_transform on the instance to encode the 'Status' column
        encoded_status = Encoder.fit_transform(features_df[["Brand Title"]])
          
        # Get the names of the encoded categories (excluding the dropped one)
        encoded_category_names = Encoder.get_feature_names_out(["Brand Title"])
          
        # Create a DataFrame from the encoded status data with appropriate column names
        encoded_status_df = pd.DataFrame(encoded_status, columns=encoded_category_names, index=features_df.index)
          
        # Drop the original 'Status' column from the features DataFrame
        features_df = features_df.drop("Brand Title", axis=1)
          
        # Concatenate the original features (without 'Status') and the encoded status columns
        data_encoded = pd.concat([features_df, encoded_status_df], axis=1)
    else:
        data_encoded = df.drop("Brand Title", axis=1)

    print(data_encoded.head())
    return data_encoded

encode_brands(data_encoded, encode = False)



"""Here I want to see correlations as a foundation



*   We detect that Price and Fees have a correlation of 1!
*   Price and Favourites and Price and Promotion have a positive correlation of 0.13
*   Also Favourites and Promotion have a positive correlation of 0.07




"""

#Visualization of Features


data_encoded['Promoted'] = data_encoded['Promoted'].astype(int)

# Select only numerical columns for the correlation heatmap
# Exclude 'Title' as it's text and not suitable for direct correlation
numerical_cols = [
    'Price', 'Favourites', 'Promoted', 'Fees', 'Status_Neu',
    'Status_Neu, mit Etikett', 'Status_Sehr gut', 'Status_Zufriedenstellend',"Time_Online_H","Favourites_per_hour"
]
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

"""Here I want to look into it more in detail"""

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

# Convert 'Price' and 'Fees' columns to numeric
data_encoded['Price'] = pd.to_numeric(data_encoded['Price'], errors='coerce')
data_encoded['Fees'] = pd.to_numeric(data_encoded['Fees'], errors='coerce')


print("\nGenerating Relationship Plots for Price/Fees and Favourites...")
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

print("Plots generated successfully!")

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




print("\nGenerating Relationship Plots for Price/Fees and Favourites...")
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