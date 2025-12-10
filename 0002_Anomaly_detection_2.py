import numpy as np

#%%   Dataset known substances
import pandas as pd
import os

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

normal_CLASSES =[
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
 # 'POTABLE_WATER',
 # 'POTASSIUM_NITRATE',
 # 'SODIUM_CHLORIDE',
 # 'SODIUM_HYDROXIDE',
 # 'SODIUM_HYPOCHLORITE'
]
# metto da parte un subset di sostanze che considerero come
# anomalie
anomaly_CLASSES =[
 # 'ACETIC_ACID',
 # 'ACETONE',
 # 'AMMONIA',
 # 'CALCIUM_NITRATE',
 # 'ETHANOL',
 # 'FORMIC_ACID',
 # 'HYDROCHLORIC_ACID',
 # 'HYDROGEN_PEROXIDE',
 # 'NELSEN',
 # 'PHOSPHORIC_ACID',
 'POTABLE_WATER',
 'POTASSIUM_NITRATE',
 'SODIUM_CHLORIDE',
 'SODIUM_HYDROXIDE',
 'SODIUM_HYPOCHLORITE'
]

# si puo sempre fare feature selection se volete, le feature inutili
# sono inutili sia per la classificazione che per l anomaly detection

# MA, mentre per la classificazione, la ridondanza non porta
# particolari benefici, nel caso di detect di anomalie, avere più
# dati potrebbe aumnetare la ROBUSTEZZA DELL ALGORITMO perchè non è
# detto che un anomalia colpisca tutti i sensori
selected_FEATURE = [
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
# uso la stessa function di prima
# già questo vi fa capire che avrei potuto farla su un file separato e importarla
def extract_dataset_slice(
        dataset_path: str,
        substances: list,
        features: list,
        experiments: list,
        time_window: tuple
):
    """
    Extract a filtered dataset slice.

    Parameters
    ----------
    dataset_path : str
        Path to folder containing CSV files.
    substances : list of str
        List of chemical substance names (prefix of CSV filenames).
    features : list of str
        List of features to include (subset of the 10 known features).
    experiments : list of int
        List of experiment numbers (0–9).
    time_window : tuple(int, int)
        Start and end of temporal window, e.g. (200, 800).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the selected subset with original CLASS column.
    """

    start, end = time_window
    if start < 0 or end > 1200 or start >= end:
        raise ValueError("Time window must be within 0–1200 and valid (start < end).")

    # Final storage
    collected = []

    # Iterate over desired substances
    for sub in substances:
        for exp in experiments:
            filename = f"{sub}_{exp}.csv"
            filepath = os.path.join(dataset_path, filename)

            if not os.path.exists(filepath):
                print(f"WARNING: file not found → {filename}")
                continue

            # Load dataset
            df = pd.read_csv(filepath)

            # Safety check
            if not set(features).issubset(df.columns):
                raise ValueError(f"Some requested features are missing in {filename}")

            # Slice time window
            sliced = df.iloc[start:end][features + ["CLASS"]]

            collected.append(sliced)

    if len(collected) == 0:
        raise ValueError("No data matched the request.")

    # Combine all pieces
    return pd.concat(collected, ignore_index=True)

# conservo una porzione di dati per il test
# NB  sta volta il test è diverso
# dobbiamo verificare che l'autoencoder sia in grado di ricostruire x
# non predirne la classe di appartenenza

list_experiment = [0,1,2,3,4,5,6,7,8,9]
test_experiment = [7,8,9]
train_experiment = [exp for exp in list_experiment if exp not in test_experiment]

time_window =(300,1000)
# applichiamola
train = extract_dataset_slice(
    dataset_path="dataset",
    substances=normal_CLASSES,
    features=selected_FEATURE,
    experiments=train_experiment,
    time_window=time_window
)
test = extract_dataset_slice(
    dataset_path="dataset",
    substances=normal_CLASSES,
    features=selected_FEATURE,
    experiments=test_experiment,
    time_window=time_window
)
anomaly = extract_dataset_slice(
    dataset_path="dataset",
    substances=anomaly_CLASSES,
    features=selected_FEATURE,
    experiments=list_experiment, # per le anomalie posso usare tutti gli esperimenti
    time_window=time_window
)


# CONSIGLIO PRATICO  a maggir raggione potete usare un file.pkl

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

x_train = train.drop(columns=["CLASS"]).to_numpy()
y_train = train[["CLASS"]].to_numpy() #non serve in realtà
x_test = test.drop(columns=["CLASS"]).to_numpy()
y_test = test[["CLASS"]].to_numpy() #non serve in realtà
x_anomaly = anomaly.drop(columns=["CLASS"]).to_numpy()
y_anomaly = anomaly[["CLASS"]].to_numpy() #non serve in realtà


#%%   encoding and normalization

scaler = MinMaxScaler()
scaler.fit(x_train)
# anche qui la lable non è che servino cosi tanto
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
feature_encoder = LabelEncoder()
feature_encoder.fit(train.columns.to_numpy())

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

label_encoder_anomaly = LabelEncoder()
# label_encoder_anomaly.fit(y_anomaly)

#%% AUTOENCODER: build, train, evaluate

# ricordo che autoencoder è una rete neurale dove la label sono le x stesse
# il dato si espande (opzionale) e poi si comprime (obbligatorio) nel buttlenek
# per essere poi ricostruito alla sua size iniziale
from tensorflow import keras

input_dim = x_train.shape[1]
encoding_dim = 16   # bottleneck size (modify to show different behaviours)

# ----- Autoencoder Model -----
input_layer = keras.layers.Input(shape=(input_dim,))
encoded = keras.layers.Dense(64, activation="relu")(input_layer)
encoded = keras.layers.Dense(32, activation="relu")(encoded)
bottleneck = keras.layers.Dense(encoding_dim, activation="relu")(encoded)

decoded = keras.layers.Dense(32, activation="relu")(bottleneck)
decoded = keras.layers.Dense(64, activation="relu")(decoded)
decoded = keras.layers.Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)
# ______________________ fine settaggio iperparametri
autoencoder.summary()

# ----- Training -----
history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)


#%% RECONSTRUCTION ERRORS

# Normal known test data
x_test_pred = autoencoder.predict(x_test)
mse_test = np.mean(np.square(x_test - x_test_pred), axis=1)

# Anomalous data
x_anom_pred = autoencoder.predict(x_anomaly)
mse_anom = np.mean(np.square(x_anomaly - x_anom_pred), axis=1)

# le anomalie generano un errore di ricostruzioni SIGNIFICATIVIAMENTE MAGGIORE
# Facciamo un plot per vederlo meglio
#%% Scatter plot — Normal vs Anomaly in 2x1 subplot
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(12,8))

# Normal samples
axs[0].scatter(range(len(mse_test)), mse_test, s=10, alpha=0.7)
axs[0].axhline(np.mean(mse_test) + 3*np.std(mse_test),
               color="red", linestyle="--", label="Threshold")
axs[0].set_title("Scatter — Reconstruction Error (NORMAL)")
axs[0].set_ylabel("MSE")
axs[0].legend()
axs[0].grid(True)

# Anomaly samples
axs[1].scatter(range(len(mse_anom)), mse_anom, s=10, alpha=0.7, color="red")
axs[1].axhline(np.mean(mse_test) + 3*np.std(mse_test),
               color="black", linestyle="--",
               label="Normal-data Threshold")
axs[1].set_title("Scatter — Reconstruction Error (ANOMALY)")
axs[1].set_ylabel("MSE")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

#%% Print some indicative values

print("Normal Test Data:")
print("  mean error  :", np.mean(mse_test))
print("  std error   :", np.std(mse_test))

print("\nAnomaly Data:")
print("  mean error  :", np.mean(mse_anom))
print("  std error   :", np.std(mse_anom))
