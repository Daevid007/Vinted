# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:32:18 2025

@author: David
"""
#Git Commands:
#git add . # stages all changes in the current directory
#git commit -m "Descriptive message about the changes I made"
#git push origin main

#git pull origin main    # downloading all changes made online (by others) to remote machine 

#git checkout -b my_new_feature   # for creating a new branch
#git push origin my_new_feature   #pushing the branch

#git branch  #checking current branch
#git branch -a   #seeing and checking all branches
#git switch <branch_name>

import requests 
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from pyVinted import Vinted
import pandas as pd
import datetime
import json



#This is the german link: 
#https://www.vinted.de/catalog?catalog[]=79&catalog_from=0&brand_ids[]=53&page=1

#Load parameter options as a dictionary
#This json/dic hast to be filled with data this is just the draft for structure
json_file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/search_parameters_overview.json"
with open(json_file_path, 'r') as file:
    my_dictionary = json.load(file)
    
vinted = Vinted()
items = vinted.items.search("https://www.vinted.de/catalog?catalog[]=79&catalog_from=0&brand_ids[]=53&page=1",10,1)
item1 = items[0]

#-----------------------------------------------------------------------------------------------------------------

#I can add parameters like search_text etc in the url string manually or with a function like the following to expand the capabilities.

def SearchVintedDraft(order = "newest_first",price_to = "60",currency = "EUR",text = "Adidas-Vintage"):
    """
    Function to search on Vinted after specified search criteria, returning a list of items
    """
    
    string = "https://www.vinted.de/vetement?order="+order+"&price_to="+price_to+"&currency="+currency+"&search_text="+text
    return vinted.items.search(string,10,1)



def showproperties():
    """
    Function to show the raw data of one item searched, returning a dictionary
    Purpose only is looking stuff up
    """
    return vinted.items.search("https://www.vinted.de/vetement?order=newest_first&price_to=60&currency=EUR&search_text=Adidas-Vintage",1,1)[0].raw_data

def save_data_parquet(name, data):
    """
    Creates a new data parquet to store the data of found items
    """
    parquet_file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet"
    data.to_parquet(parquet_file_path, index=False, compression='snappy')
    print(f"\nDataFrame successfully saved to {parquet_file_path}")


def load_data_parquet(name):
    """
    Loads a specified data_parquet into a dataframe and returns it
    """
    return pd.read_parquet("C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet")


def add_data_to_parquet(name,data):
    """
    Adds search data to an existing data parquet, saves this parquet and returns the merged DataFrame
    Also removes duplicates
    """
    tmp = pd.read_parquet("C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet")
    data = pd.concat([tmp,data])
    parquet_file_path = "C:/Users/David/OneDrive - fs-students.de/Vinted/Data/"+name+"_data_parquet"
    data = data.drop_duplicates()
    data.to_parquet(parquet_file_path, index=False, compression='snappy')
    print(f"\nDataFrame successfully saved to {parquet_file_path}")
    return data


    
    
#------------------------------------------------------------------------------------------------------------------
#to work with the pictures
items_searched = SearchVintedDraft() 

df = pd.DataFrame({"ID":[],
                   "Title": [],
                   "Price":[],
                   "Favourites":[],
                   "Link":[],
                   "Brand Title":[],
                   "Promoted":[],
                   "Status":[],
                   "Views":[],
                   "Dates":[]})
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
Views = [item.raw_data["view_count"] for item in items_searched]
Dates = [datetime.datetime.fromtimestamp(item.raw_timestamp) for item in items_searched]

df["ID"] = IDs
df["Title"] = titles
df["Price"] = prices
df["Favourites"] = favourites
df["Link"] = Links
df["Brand Title"] = Brands
df["Promoted"] = Promoted
df["Status"] = Status
df["Views"] = Views
df["Dates"] = Dates


#------------------------------------------------------------------------------------------------------------------
#Show pictures of the items 
for each in items_searched:
    # URL of the image
    url = each.photo
    
    # Fetch image from URL
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Display image
    plt.imshow(img)
    plt.title(each.title)
    plt.axis('off')  # Hide axes
    plt.show()


#Next step is to adding to the data via different criteria

#------------------------------------------------------------------------------------------------------------------

#Adding all adittional data to the test_data_parquet
data = add_data_to_parquet("test", df)
