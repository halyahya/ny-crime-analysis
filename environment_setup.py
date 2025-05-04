#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 23:25:37 2025

@author: hanaalyahya
"""

"""
Environment Setup Instructions:

Before running, install required packages.

If using Conda:
    conda install -c conda-forge xgboost

Or using pip:
    pip install xgboost pandas numpy tqdm matplotlib seaborn scikit-learn
"""

# ---- Import required libraries ----
import pandas as pd
import zipfile
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

# Machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# ---- Set up visualization style ----
# === Theme Colors ===
BACKGROUND_COLOR = "#1f1f2e"
TEXT_COLOR = "white"
ACCENT_COLORS = ["#60A9FF", "#8CB4FF", "#B3A3FF", "#D187FF"]

# === Apply Universal Style ===
plt.style.use("dark_background")
sns.set_style("darkgrid", {"axes.facecolor": BACKGROUND_COLOR})
plt.rcParams.update({
    "axes.facecolor": BACKGROUND_COLOR,
    "figure.facecolor": BACKGROUND_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "font.size": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

# ---- Create visualizations directory ----
os.makedirs("visualizations", exist_ok=True)
