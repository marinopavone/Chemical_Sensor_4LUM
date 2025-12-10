import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"  # <-- your folder


def plot_feature(csv_filename, feature_name):
    # Build full path
    file_path = os.path.join(DATASET_DIR, csv_filename)

    # Check file exists
    if not os.path.isfile(file_path):
        print(f"ERROR: File '{csv_filename}' was not found in '{DATASET_DIR}'")
        return

    # Load CSV
    df = pd.read_csv(file_path)

    # Remove the CLASS column (always last)
    df = df.iloc[:, :-1]

    # Check feature exists
    if feature_name not in df.columns:
        print("ERROR: Feature not found.")
        print("Available features:")
        for col in df.columns:
            print("  -", col)
        return

    # Plot the selected feature
    plt.figure(figsize=(10, 4))
    plt.plot(df[feature_name])
    plt.title(f"{feature_name} from {csv_filename}")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


# -------------------------------
# Example usage:
# -------------------------------

# Plot from file "ACETIC_ACID_0.csv" the feature "OFFCHIP_PLATINUM_78kHz_IN-PHASE"
plot_feature("ACETIC_ACID_0.csv", "OFFCHIP_GOLD_78kHz_IN-PHASE")
