import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import os

# --------------------------------------------------------------
# Get the directory where the script is located and changing the working directory
# --------------------------------------------------------------

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"]==1]
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, x = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()
    
for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, x = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()
    # display(subset.head(2))


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20,5)
mpl.rcParams['figure.dpi'] =  100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat '")

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# -------------------------------------------------------------- 