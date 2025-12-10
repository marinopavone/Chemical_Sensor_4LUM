import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "dataset"  # your folder

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

# per comprendere gli algoritmi di clustering, vediamo cosa succede se
# facciamo un plot a 2 dimensioni dei dati di una singola classe
#
# ricordiamoci di mostrare solo i dati "stabilizzati"
transient_cut = 300

def plot_two_features_for_class(class_name, feature_x, feature_y):
    """
    Plot a 2D scatter of two features for all samples of a given chemical class.
    """
    all_data = []
    dataset_path=DATASET_DIR
    # Load ONLY files that start with the class name
    transient_cut = 300
    for file in os.listdir(dataset_path):
        if file.startswith(class_name) and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dataset_path, file))
            # all_data.append(df)
            all_data.append(df[transient_cut:])

    if not all_data:
        raise ValueError(f"No files found for class '{class_name}'")

    # Concatenate all samples for the class
    data = pd.concat(all_data, ignore_index=True)

    # Safety checks
    if feature_x not in data.columns:
        raise ValueError(f"Feature '{feature_x}' not found in dataset")
    if feature_y not in data.columns:
        raise ValueError(f"Feature '{feature_y}' not found in dataset")

    # Extract the two features
    x = data[feature_x].values
    y = data[feature_y].values
    x = data[feature_x].values[500:]
    y = data[feature_y].values[500:]

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, s=20, alpha=0.7)
    plt.title(f"{class_name} — 2D Feature Scatter")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_two_features_for_class(
    class_name="NELSEN",
    feature_x="OFFCHIP_PLATINUM_78kHz_IN-PHASE",
    feature_y="OFFCHIP_GOLD_200Hz_QUADRATURE"
)


import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# per capire cosa significa fare clustering su una distribuzione di dati, aggiungiamo
# altre classi

# nonstante stiamo usando solo 2 feature delle 10 disponibili, è gia visivamente
# chiaro che si formano cluster separabili in base alle classi
#
# le classi più difficili da classificare infatti, sono quelle dove
# "i colori si mischiano un po"
def plot_features_multi_class(dataset_path, class_list, feature_x, feature_y):
    """
    Plot a 2D scatter of two features for multiple chemical classes.

    Parameters:
        dataset_path : str
            Path to the dataset folder containing CSV files.
        class_list : list of str
            List of chemical class names to include.
        feature_x : str
            Column name for x-axis.
        feature_y : str
            Column name for y-axis.
    """

    plt.figure(figsize=(10, 8))

    # Use a color palette from seaborn for consistent colors
    palette = sns.color_palette("tab10", n_colors=len(class_list))

    for idx, class_name in enumerate(class_list):
        all_data = []

        # Load all CSV files for the class
        for file in os.listdir(dataset_path):
            if file.startswith(class_name) and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(dataset_path, file))
                # all_data.append(df)
                all_data.append(df[transient_cut:])

        if not all_data:
            print(f"WARNING: No files found for class '{class_name}'")
            continue

        # Concatenate all samples
        data = pd.concat(all_data, ignore_index=True)

        # Safety checks
        if feature_x not in data.columns or feature_y not in data.columns:
            print(f"WARNING: Feature '{feature_x}' or '{feature_y}' not found for class '{class_name}'")
            continue

        # Plot
        plt.scatter(
            data[feature_x],
            data[feature_y],
            s=20,
            alpha=0.7,
            color=palette[idx],
            label=class_name
        )

    plt.title(f"2D Feature Plot: '{feature_x}' vs '{feature_y}'")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

classes_to_plot = [
    "NELSEN",
    "ACETIC_ACID",
    "POTASSIUM_NITRATE",
    "HYDROCHLORIC_ACID",
    "ETHANOL"
]

plot_features_multi_class(
    dataset_path="dataset",
    class_list=classes_to_plot,
    feature_x="OFFCHIP_PLATINUM_78kHz_IN-PHASE",
    feature_y="OFFCHIP_GOLD_200Hz_QUADRATURE"
)