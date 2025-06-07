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



import requests 
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from pyVinted import Vinted



vinted = Vinted()
items = vinted.items.search("https://www.vinted.fr/vetement?order=newest_first&price_to=60&currency=EUR&search_text=BVB",10,1)
item1 = items[0]
#title
item1.title
#id
item1.id
#photo url
item1.photo
#brand
item1.brand_title
#price
item1.price
#url
item1.url
#currency
item1.currency

#-----------------------------------------------------------------------------------------------------------------

#I can add parameterslike search_text etc in the url string manually or with a function like the following to expand the capabilities.

def SearchVintedDraft(order = "newest_first",price_to = "60",currency = "EUR",text = "BVB Trikot"):
    string = "https://www.vinted.fr/vetement?order="+order+"&price_to="+price_to+"&currency="+currency+"&search_text="+text
    return vinted.items.search(string,20,1)

items_searched = SearchVintedDraft()

for each in items_searched:
    print(each.title,"price:",each.price,"brand",each.brand_title)


def showproperties():
   return vinted.items.search("https://www.vinted.fr/vetement?order=newest_first&price_to=60&currency=EUR&search_text=BVB",1,1)[0].raw_data

    
def getlikes():
    return vinted.items.search("https://www.vinted.fr/vetement?order=newest_first&price_to=60&currency=EUR&search_text=BVB",1,1)[0].raw_data["favourite_count"]
#------------------------------------------------------------------------------------------------------------------
#to work with the pictures

properties = showproperties()

favs = getlikes()

for each in items_searched:
    # URL of the image
    url = each.photo

    # Fetch image from URL
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Display image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
