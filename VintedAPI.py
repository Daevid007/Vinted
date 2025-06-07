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


vinted = Vinted()
items = vinted.items.search("https://www.vinted.de/vetement?order=newest_first&price_to=60&currency=EUR&search_text=BVB",10,1)
item1 = items[0]

#-----------------------------------------------------------------------------------------------------------------

#I can add parameters like search_text etc in the url string manually or with a function like the following to expand the capabilities.

def SearchVintedDraft(order = "newest_first",price_to = "60",currency = "EUR",text = "BVB Trikot"):
    string = "https://www.vinted.de/vetement?order="+order+"&price_to="+price_to+"&currency="+currency+"&search_text="+text
    return vinted.items.search(string,10,1)


def showproperties():
   return vinted.items.search("https://www.vinted.de/vetement?order=newest_first&price_to=60&currency=EUR&search_text=BVB",1,1)[0].raw_data

    
def getlikes():
    return vinted.items.search("https://www.vinted.de/vetement?order=newest_first&price_to=60&currency=EUR&search_text=BVB",1,1)[0].raw_data["favourite_count"]
#------------------------------------------------------------------------------------------------------------------
#to work with the pictures
items_searched = SearchVintedDraft() 

df = pd.DataFrame({"Title": [],
                   "Price":[],
                   "Favourites":[],
                   "Link":[],
                   "Brand Title":[],
                   "Promoted":[],
                   "Status":[],
                   "Views":[],
                   "Dates":[]})
#------------------------------------------------------------------------------------------------------------------


titles = [item.title for item in items_searched]
prices = [item.price for item in items_searched]
favourites = [item.raw_data["favourite_count"] for item in items_searched]
Links = [item.url for item in items_searched]
Brands = [item.brand_title for item in items_searched]    
Promoted = [item.raw_data["promoted"] for item in items_searched]
Status = [item.raw_data["status"] for item in items_searched]
Views = [item.raw_data["view_count"] for item in items_searched]
Dates = [datetime.datetime.fromtimestamp(item.raw_timestamp) for item in items_searched]

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
