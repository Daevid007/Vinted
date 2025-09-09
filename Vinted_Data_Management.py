# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:04:28 2025

@author: david
"""

import numpy as np
import pandas as pd
import Vinted_Definitions as vd

#What File?
name = "test2"

#Creating Files
create = True
if create == True:
    vd.create_data_parquet(name)

#Deleting Files
delete = False
if delete == True:
    vd.delete_data_parquet(name)

#Loading Files
load = True
if load == True:
    loaded_data = vd.load_data_parquet(name)
    
#Collecting Data
collect = False
if collect == True:
    vd.collect_data(parameters_text = ["Adidas-Vintage","Nike-Vintage","Reebook-Vintage","Puma-Vintage","WRSTBHVR","Kappa-Vintage",    "Puma Archive",
    "Reebok Classics",
    "Fila Vintage",
    "Champion Reverse Weave",
    "Ellesse Heritage",
    "Kappa Authentic",
    "Umbro Classics",
    "Diadora Heritage",
    "Lacoste Sport Vintage",
    "Levi's Vintage Clothing",
    "Carhartt WIP",
    "Patagonia Retro",
    "The North Face Vintage",
    "Columbia Sportswear Vintage",
    "Tommy Hilfiger Classics",
    "Ralph Lauren Polo Sport",
    "Converse Vintage",
    "Vans Classics",
    "New Balance Heritage",
    "Asics Onitsuka Tiger",
    "Starter Vintage",
    "Russell Athletic Vintage",
    "Nautica Competition",
    "Dickies Vintage",
    "Wrangler Retro",
    "Lee Jeans Vintage",
    "Timberland Heritage",
    "Jordan Brand Retro"])
    
#Check_cleaning
check = False
if check == True:
    vd.check_clean_data(name)