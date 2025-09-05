# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:08:14 2025

@author: david
"""

import numpy as np
import pandas as pd
import Vinted_Definitions as vd

#What file?
name = "test"
data = vd.load_data_parquet(name)

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
    
