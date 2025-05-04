#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:25:29 2025

@author: hanaalyahya
"""


import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from environment_setup import BACKGROUND_COLOR

def apply_styled_background(ax, fig, title_text=None):
    """
    Apply a styled background with rounded corners and shadow effect to the plot.
    """
    rect_shadow = plt.Rectangle((0.01, -0.01), 1, 1, transform=fig.transFigure,
                               facecolor="#000000", alpha=0.15, linewidth=0, zorder=-2, clip_on=False)
    rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                        facecolor=BACKGROUND_COLOR, edgecolor="#D187FF",
                        linewidth=1.5, zorder=-1, clip_on=False)
    fig.patches.extend([rect_shadow, rect])

    if title_text:
        title = ax.set_title(title_text, fontsize=16, color='white', pad=20)
        title.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

def unzip_file(zip_name, year, has_subfolder=False):
    """
    Unzip a data file and return the path to the extracted contents.
    """
    zip_path = f"data/{zip_name}" 
    extract_path = f"tmp_data/{year}/"
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    if has_subfolder:
        return os.path.join(extract_path, f"{zip_name.replace('.zip','')}/NY/")
    else:
        return extract_path

def load_year_data(folder):
    """
    Load and merge crime data files for a specific year.
    """
    inc = pd.read_csv(folder + "NIBRS_incident.csv", low_memory=False)
    off = pd.read_csv(folder + "NIBRS_OFFENSE.csv")
    typ = pd.read_csv(folder + "NIBRS_OFFENSE_TYPE.csv")

    df = pd.merge(off, inc, on="incident_id", how="inner")
    df = pd.merge(df, typ, on="offense_code", how="left")

    return df

def simplify_category(cat):
    """
    Group detailed crime categories into broader categories.
    """
    if "Assault" in cat or "Sex Offense" in cat:
        return "Assault/Sex"
    elif "Theft" in cat or "Fraud" in cat or "Burglary" in cat or "Embezzlement" in cat:
        return "Theft/Fraud"
    elif "Drug" in cat or "Weapon" in cat:
        return "Drug/Weapon"
    elif "Vandalism" in cat or "Arson" in cat:
        return "Property Damage"
    elif "Homicide" in cat or "Kidnapping" in cat or "Robbery" in cat:
        return "Violent"
    else:
        return "Other"
