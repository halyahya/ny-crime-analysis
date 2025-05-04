#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:34:16 2025

@author: hanaalyahya
"""


from environment_setup import *
import pandas as pd
from tqdm import tqdm

# ---- Load cleaned data ----
df = pd.read_pickle("cleaned_data.pkl")
print("Loaded cleaned data")

# ---- Remove rows with missing values ----
df = df.dropna(subset=["incident_date", "location_id", "incident_id"])

# ---- Sort data chronologically ----
df = df.sort_values("incident_date")
df.set_index("incident_date", inplace=True)

# ---- Initialize columns for smart features ----
df["crime_count_7d_location"] = 0
df["crime_count_30d_location"] = 0
df["crime_count_30d_hour"] = 0

# ---- Calculate rolling crime counts ----
print("Calculating rolling crime counts... (~2–4 min)")

for i in tqdm(range(len(df))):
    row = df.iloc[i]
    t, loc, hour = row.name, row["location_id"], row["incident_hour"]

    past_7d = df.loc[t - pd.Timedelta(days=7):t - pd.Timedelta(seconds=1)]
    past_30d = df.loc[t - pd.Timedelta(days=30):t - pd.Timedelta(seconds=1)]

    df.iloc[i, df.columns.get_loc("crime_count_7d_location")] = past_7d[past_7d["location_id"] == loc].shape[0]
    df.iloc[i, df.columns.get_loc("crime_count_30d_location")] = past_30d[past_30d["location_id"] == loc].shape[0]
    df.iloc[i, df.columns.get_loc("crime_count_30d_hour")] = past_30d[past_30d["incident_hour"] == hour].shape[0]

# ---- Reset index ----
df.reset_index(inplace=True)
print("Smart features added!")

# ---- Save enhanced data ----
df.to_csv("crime_smart_features.csv", index=False)
print("Saved smart-feature-enhanced data to crime_smart_features.csv!")

# ---- Output sample ----
print(df[["incident_date", "location_id", "incident_hour", "crime_count_7d_location",
          "crime_count_30d_location", "crime_count_30d_hour", "crime_group"]].head())