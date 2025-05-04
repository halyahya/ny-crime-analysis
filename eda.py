#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:47:14 2025

@author: hanaalyahya
"""


from environment_setup import *
from utilities import apply_styled_background
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Load the smart-feature-enhanced data ----
df = pd.read_csv("crime_smart_features.csv")
print("Loaded smart-feature-enhanced data")

# ---- 1. Crime by Category ----
crime_counts = df['crime_group'].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
crime_counts.plot(kind='bar', color=ACCENT_COLORS[0], ax=ax)
apply_styled_background(ax, fig, "Crime Distribution by Category")
ax.set_xlabel("Crime Category")
ax.set_ylabel("Number of Incidents")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("visualizations/crime_by_category.png", facecolor=BACKGROUND_COLOR)
plt.show()

# ---- 2. Crime by Hour Group ----
hour_ranges = ["00–05", "06–11", "12–17", "18–23"]
hour_labels = ["Early Morning", "Morning", "Afternoon", "Evening"]
hour_counts = df["hour_group"].value_counts().reindex(hour_labels)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(hour_labels, hour_counts, color=ACCENT_COLORS[1])
apply_styled_background(ax, fig, "Crime Count by Time of Day")
ax.set_xlabel("Hour Group")
ax.set_ylabel("Number of Crimes")
for bar, label in zip(bars, hour_ranges):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 10, label,
            ha='center', va='bottom', fontsize=10, color=TEXT_COLOR)
plt.tight_layout()
plt.savefig("visualizations/crime_by_hour_group.png", facecolor=BACKGROUND_COLOR)
plt.show()

# ---- 3. Crime by Day of Week ----
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_counts = df["day_of_week"].value_counts().reindex(dow_order)

fig, ax = plt.subplots(figsize=(8, 5))
dow_counts.plot(kind="bar", color=ACCENT_COLORS[2], ax=ax)
apply_styled_background(ax, fig, "Crime Count by Day of Week")
ax.set_xlabel("Day")
ax.set_ylabel("Number of Crimes")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("visualizations/crime_by_day.png", facecolor=BACKGROUND_COLOR)
plt.show()

# ---- 4. Crime by Season ----
season_colors = {
    "Winter": "#60A9FF",
    "Spring": "#8CB4FF",
    "Summer": "#B3A3FF",
    "Fall": "#D187FF"
}
season_counts = df["season"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"])
colors = [season_colors[season] for season in season_counts.index]

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(season_counts.index, season_counts.values, color=colors)
apply_styled_background(ax, fig, "Crime Count by Season")
ax.set_xlabel("Season")
ax.set_ylabel("Number of Crimes")
plt.tight_layout()
plt.savefig("visualizations/crime_by_season.png", facecolor=BACKGROUND_COLOR)
plt.show()

# ---- 5. Heatmap: Crime Category vs. Hour ----
heatmap_data = pd.crosstab(df["hour"], df["crime_group"])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data.T, cmap="coolwarm", linewidths=0.3, linecolor=BACKGROUND_COLOR,
            cbar_kws={'label': 'Crime Count'}, ax=ax)
apply_styled_background(ax, fig, "Crime Category by Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("Crime Type")
plt.tight_layout()
plt.savefig("visualizations/crime_heatmap.png", facecolor=BACKGROUND_COLOR)
plt.show()

# ---- 6. Weekday vs Weekend Comparison ----
grouped = df.groupby(["is_weekend", "crime_group"]).size().unstack().fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
grouped.T.plot(kind="bar", color=[ACCENT_COLORS[0], ACCENT_COLORS[3]], ax=ax)
apply_styled_background(ax, fig, "Weekday vs. Weekend Crime Count by Category")
ax.set_xlabel("Crime Category")
ax.set_ylabel("Number of Crimes")
ax.legend(["Weekday", "Weekend"], title="Day Type")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("visualizations/weekday_vs_weekend_crime.png", facecolor=BACKGROUND_COLOR)
plt.show()