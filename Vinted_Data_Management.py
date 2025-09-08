# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:04:28 2025

@author: david
"""

import numpy as np
import pandas as pd
import Vinted_Definitions as vd

#What File?
name = "test"

#Creating Files
create = False
if create == True:
    vd.create_data_parquet(name)

#Deleting Files
delete = False
if delete == True:
    vd.delete_data_parquet(name)

#Loading Files
load = False
if load == True:
    loaded_data = vd.load_data_parquet(name)
    
#Collecting Data
collect = True
if collect == True:
    vd.collect_data(parameters_text = ["Adidas-Vintage","Nike-Vintage","Reebook-Vintage","Puma-Vintage","WRSTBHVR","Kappa-Vintage"])
    
#Check_cleaning
check = False
if check == True:
    vd.check_clean_data(name)