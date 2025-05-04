#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:39:11 2025

@author: hanaalyahya
"""


from environment_setup import *
from utilities import simplify_category
import pandas as pd

# ---- Load combined data ----
df = pd.read_pickle("combined_data.pkl")
print("Loaded combined data")

# ---- Convert date and extract time features ----
df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
df["hour"] = df["incident_hour"]
df["day_of_week"] = df["incident_date"].dt.day_name()
df["month"] = df["incident_date"].dt.month

# ---- Create season feature ----
df["season"] = df["month"] % 12 // 3 + 1
df["season"] = df["season"].map({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})

# ---- Create hour groups for visualization ----
hour_bins = [0, 6, 12, 18, 24]
hour_labels = ["Early Morning", "Morning", "Afternoon", "Evening"]
df["hour_group"] = pd.cut(df["hour"], bins=hour_bins, labels=hour_labels, right=False)

# ---- Weekday/weekend indicator ----
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

# ---- Crime category grouping ----
df["crime_group"] = df["offense_category_name"].apply(simplify_category)

# ---- Sample data to avoid memory issues ----
df = df.sample(frac=0.2, random_state=42)

# ---- Output ----
print("\nCrime Group Distribution:")
print(df["crime_group"].value_counts())

print("\nSample data with new features:")
print(df[["incident_date", "hour", "day_of_week", "season", "hour_group", "is_weekend", "crime_group"]].head())

# ---- Save cleaned data ----
df.to_pickle("cleaned_data.pkl")
print("Cleaned data saved to cleaned_data.pkl")