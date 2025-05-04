#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:32:15 2025

@author: hanaalyahya
"""


from environment_setup import *
from utilities import unzip_file, load_year_data
import pandas as pd

# ---- Unzip and define paths ----
paths = {
    "2021": unzip_file("NY-2021.zip", "2021"),
    "2022": unzip_file("NY-2022.zip", "2022"),
    "2023": unzip_file("NY-2023.zip", "2023", has_subfolder=True),
}

# ---- Load and combine data ----
dfs = [load_year_data(path) for path in paths.values()]
df = pd.concat(dfs, ignore_index=True)

print("Combined data shape:", df.shape)

# ---- Display a sample ----
print(df.head())



# ---- Save combined data ----
df.to_pickle("combined_data.pkl")
print("Data saved to combined_data.pkl")
