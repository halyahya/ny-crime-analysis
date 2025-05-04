#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 21:57:56 2025

@author: hanaalyahya
"""


from environment_setup import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ---- Load the smart-feature-enhanced data ----
df = pd.read_csv("crime_smart_features.csv")
print("Loaded smart-feature-enhanced data")

# ---- Sample fixed number of rows ----
df = df.sample(n=50000, random_state=42)

# ---- Remove missing target values ----
df.dropna(subset=["crime_group"], inplace=True)

# ---- Additional feature engineering ----
df["hour_day_combo"] = df["hour"].astype(str) + "_" + df["day_of_week"]
df["location_crime_total"] = df.groupby("location_id")["incident_id"].transform("count")

# ---- Select features ----
features = df[[
    "hour", "day_of_week", "season", "attempt_complete_flag", "crime_against", "location_id",
    "crime_count_7d_location", "crime_count_30d_location", "crime_count_30d_hour",
    "hour_day_combo", "location_crime_total"
]]

# ---- Target variable ----
target = df["crime_group"]

# ---- Encode categorical features ----
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = encoder.fit_transform(features)

# ---- Encode target labels ----
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target)

# ---- Split data ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# ---- Save encoders and data splits for reuse ----
import joblib

joblib.dump(encoder, "encoder.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")
joblib.dump((X_train, X_test, y_train, y_test), "train_test_data.joblib")

print("Encoders and data splits saved for modeling")
