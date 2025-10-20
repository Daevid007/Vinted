# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:04:28 2025

@author: david
"""

import numpy as np
import pandas as pd
import Vinted_Definitions as vd

#------------------------------------------------------Adjustable-----------------------------------------------------
#IMG
name_img = "test2"
link_img = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name_img+"_data_img.h5"
name_img_parquet = "test2"
link_img_parquet = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name_img_parquet+"_data_parquet"




#File
name = "test"
name_2 = "test2"

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

#Pathes
git_path = "https://github.com/Daevid007/Vinted/blob/main/Data/"+name+"_data_parquet?raw=true"
git_path_2 = "https://github.com/Daevid007/Vinted/blob/main/Data/"+name_2+"_data_parquet?raw=true"

local_path_1 = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"
local_path_2 = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name_2+"_data_parquet"

#r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"

#Collecting Data
list1 = ["Adidas-Vintage","Nike-Vintage","Reebook-Vintage","Puma-Vintage","WRSTBHVR","Kappa-Vintage", "Puma Archive",
"Reebok Classics", "Fila Vintage", "Champion Reverse Weave", "Ellesse Heritage", "Kappa Authentic", "Umbro Classics",
"Diadora Heritage", "Lacoste Sport Vintage", "Levi's Vintage Clothing", "Carhartt WIP", "Patagonia Retro", "The North Face Vintage",
"Columbia Sportswear Vintage", "Tommy Hilfiger Classics", "Ralph Lauren Polo Sport", "Converse Vintage", "Vans Classics",
"New Balance Heritage", "Asics Onitsuka Tiger", "Starter Vintage", "Russell Athletic Vintage", "Nautica Competition",
"Dickies Vintage", "Wrangler Retro", "Lee Vintage", "Timberland Heritage", "Jordan Brand Retro"]

list2 = ["Adidas-Vintage"]
#--------------------------------------------------------------------------------------------------------------------

#Creating Files
create = False
if create == True:
    vd.create_data_parquet(name,local_path_1)


#Deleting Files
delete = False
if delete == True:
    vd.delete_data_parquet(name,local_path_1)


#Loading Files
load = True
if load == True:
    loaded_data = vd.load_data_parquet(name,local_path_1)
    
    
collect = False
if collect == True:
    collected_data = vd.collect_data(parameters_text = list1,parquet_file_path=local_path_1,name = name)
    
    
#Check_cleaning
check = False
if check == True:
    vd.check_clean_data(name,parquet_file_path=git_path)
    
#Saving Files
save = False
if save == True:
    vd.save_data_parquet(name_2, data, parquet_file_path = local_path_2)
    
#Loading img df
load_img = True
if load_img == True:
    encoded_img_data = vd.load_img_data(file_path=link_img, parquet_file_path=link_img_parquet)
    
    
#Encoding and Storing imgs
img = True
if img == True:
    vd.store_img_data(df = vd.add_image_array_data(df_unloaded = loaded_data.loc[800:900,:], df_loaded = encoded_img_data), link = link_img, parquet_file_path=link_img_parquet)
    

