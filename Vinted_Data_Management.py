# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:04:28 2025

@author: david
"""

import numpy as np
import pandas as pd
import Vinted_Definitions as vd

#------------------------------------------------------Adjustable-----------------------------------------------------

#File
name = "test"

#Pathes
git_path = "https://github.com/Daevid007/Vinted/blob/main/Data/"+name+"_data_parquet?raw=true"
local_path_1 = r"C:\Users\david\OneDrive - fs-students.de\Vinted\Data\\"+name+"_data_parquet"
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
load = False
if load == True:
    loaded_data = vd.load_data_parquet(name,git_path)
    
    
collect = False
if collect == True:
    vd.collect_data(parameters_text = list2,parquet_file_path=local_path_1,name = name)
    
    
#Check_cleaning
check = False
if check == True:
    vd.check_clean_data(name,parquet_file_path=git_path)