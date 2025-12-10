import os
import pandas as pd
import matplotlib.pyplot as plt

# questa funzione aiuta a visualizzare la differenza tra un esperimento e l altro
# osservando quanto, tutte le feature del dataset, variano da esperimeno a esperimento
# (abbiamo 10 esperimenti qui, 10 iniezioni di inquinante nell acqua))
DATASET_DIR = "dataset"  # <-- your folder
FEATURE_NAMES = [
    "OFFCHIP_PLATINUM_78kHz_IN-PHASE",
    "OFFCHIP_GOLD_78kHz_IN-PHASE",
    "OFFCHIP_PLATINUM_200Hz_IN-PHASE",
    "OFFCHIP_PLATINUM_200Hz_QUADRATURE",
    "OFFCHIP_GOLD_200Hz_IN-PHASE",
    "OFFCHIP_GOLD_200Hz_QUADRATURE",
    "OFFCHIP_SILVER_200Hz_IN-PHASE",
    "OFFCHIP_SILVER_200Hz_QUADRATURE",
    "OFFCHIP_NICKEL_200Hz_IN-PHASE",
    "OFFCHIP_NICKEL_200Hz_QUADRATURE"
]
CLASSES_NAMES =[
 'ACETIC_ACID',
 'ACETONE',
 'AMMONIA',
 'CALCIUM_NITRATE',
 'ETHANOL',
 'FORMIC_ACID',
 'HYDROCHLORIC_ACID',
 'HYDROGEN_PEROXIDE',
 'NELSEN',
 'PHOSPHORIC_ACID',
 'POTABLE_WATER',
 'POTASSIUM_NITRATE',
 'SODIUM_CHLORIDE',
 'SODIUM_HYDROXIDE',
 'SODIUM_HYPOCHLORITE'
]
def plot_all_features_class(class_name):
    # Find all CSV files that correspond to the class
    files = [
        f for f in os.listdir(DATASET_DIR)
        if f.startswith(class_name) and f.endswith(".csv")
    ]

    if not files:
        print(f"ERROR: No CSV files found for class '{class_name}'")
        return

    # Load all files and concatenate them
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(DATASET_DIR, f))
        df = df.iloc[:, :-1]  # Remove CLASS column
        dfs.append(df)

    # One big dataframe (all measurements for that class)
    data = pd.concat(dfs, ignore_index=True)

    # Begin plotting
    fig, axs = plt.subplots(2, 5, figsize=(16, 6))
    axs = axs.flatten()

    for i, feature in enumerate(FEATURE_NAMES):
        axs[i].plot(data[feature])
        axs[i].set_title(feature, fontsize=8)
        axs[i].grid(True)

    fig.suptitle(f"All Features for Class: {class_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_all_features_class(class_name="AMMONIA")
plot_all_features_class(class_name="NELSEN")

import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"   # folder containing CSVs

FEATURE_NAMES = [
    "OFFCHIP_PLATINUM_78kHz_IN-PHASE",
    "OFFCHIP_GOLD_78kHz_IN-PHASE",
    "OFFCHIP_PLATINUM_200Hz_IN-PHASE",
    "OFFCHIP_PLATINUM_200Hz_QUADRATURE",
    "OFFCHIP_GOLD_200Hz_IN-PHASE",
    "OFFCHIP_GOLD_200Hz_QUADRATURE",
    "OFFCHIP_SILVER_200Hz_IN-PHASE",
    "OFFCHIP_SILVER_200Hz_QUADRATURE",
    "OFFCHIP_NICKEL_200Hz_IN-PHASE",
    "OFFCHIP_NICKEL_200Hz_QUADRATURE"
]
# con questa funzioe possimo osservare più nel dettaglio la singola featrure di interesse
# che varia tra i 10 esperimenti
def plot_class_feature(class_name, feature_name):
    # Find all CSV files for the class
    files = sorted(
        f for f in os.listdir(DATASET_DIR)
        if f.startswith(class_name) and f.endswith(".csv")
    )

    if not files:
        print(f"ERROR: No CSV files found for class '{class_name}'")
        return

    fig, axs = plt.subplots(2, 5, figsize=(16, 6))
    axs = axs.flatten()

    for i, f in enumerate(files[:10]):  # max 10 files
        df = pd.read_csv(os.path.join(DATASET_DIR, f))

        if feature_name not in df.columns:
            print(f"ERROR: Feature '{feature_name}' not found in file {f}")
            return

        axs[i].plot(df[feature_name])
        axs[i].set_title(f"{f}", fontsize=8)
        axs[i].grid(True)

    # If fewer than 10 files, hide extra axes
    for j in range(len(files), 10):
        axs[j].axis("off")

    fig.suptitle(f"Feature: {feature_name}   —   Class: {class_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_class_feature(class_name="AMMONIA", feature_name="OFFCHIP_NICKEL_200Hz_IN-PHASE")
plot_class_feature(class_name="AMMONIA", feature_name="OFFCHIP_SILVER_200Hz_IN-PHASE")
plot_class_feature(class_name="NELSEN", feature_name="OFFCHIP_PLATINUM_200Hz_IN-PHASE")
plot_class_feature(class_name="NELSEN", feature_name="OFFCHIP_PLATINUM_78kHz_IN-PHASE")






