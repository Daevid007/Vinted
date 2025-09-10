# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:08:14 2025

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

#--------------------------------------------------------------------------------------------------------------------

data = vd.load_data_parquet(name,parquet_file_path = git_path)

numerical_cols = [
        'Price', 'Favourites', 'Promoted', 'Fees', 'Status_Neu',
        'Status_Neu, mit Etikett', 'Status_Sehr gut', 'Status_Zufriedenstellend',
        "Time_Online_H","Favourites_per_hour"
]

encoded_data = vd.encode_cols(data)

heatmap = False
if heatmap == True:
    vd.heatmap_vinted(encoded_data,numerical_cols)
    
    
scatter_favourites = False
if scatter_favourites == True:
    vd.scatter_favourites(encoded_data, numerical_cols)
    
    
price_fees = False
if price_fees == True:
    vd.plots_price_fees(encoded_data, numerical_cols)
    
