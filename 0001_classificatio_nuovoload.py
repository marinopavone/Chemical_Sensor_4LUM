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

# se volete vedere cosa succede diminuendo il n di classi
selected_CLASSES =[
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
# se volete applicare algoritmi di feautre selection,
# sarà l'algoritmo e dare come output la variabile selected_FEATURE
# per ora possiamo escludere banalmente commentando con un #
selected_FEATURE = [
    "OFFCHIP_PLATINUM_78kHz_IN-PHASE",
    "OFFCHIP_GOLD_78kHz_IN-PHASE",
    # "OFFCHIP_PLATINUM_200Hz_IN-PHASE",
    # "OFFCHIP_PLATINUM_200Hz_QUADRATURE",
    # "OFFCHIP_GOLD_200Hz_IN-PHASE",
    "OFFCHIP_GOLD_200Hz_QUADRATURE",
    "OFFCHIP_SILVER_200Hz_IN-PHASE",
    "OFFCHIP_SILVER_200Hz_QUADRATURE",
    "OFFCHIP_NICKEL_200Hz_IN-PHASE",
    "OFFCHIP_NICKEL_200Hz_QUADRATURE"
]

# QQuesta funzione filtra il dataset, potete quindi usarla per selezionare velocemente
# classi, feature, esperimenti e anche la finestra temporale
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
# se l algoritmo funziona, sarà in grado di riconoscere sostanze
# anche se i dati vengono da un altro esperimento
list_experiment = [0,1,2,3,4,5,6,7,8,9]
test_experiment = [7,8,9]
train_experiment = [exp for exp in list_experiment if exp not in test_experiment]
# ricordo che ogni esperimento è un acquisizione di 1000 campioni di tutte e 10 le feature
# che si avvia nell esatto momento in cui si avvia l'ignezione
# tagliare i primi campioni evita di usare i dati nel transitori e considerare il
# problema come una classificazione statica
time_window =(300,1000)
# applichiamola
train = extract_dataset_slice(
    dataset_path="dataset",
    substances=selected_CLASSES,
    features=selected_FEATURE,
    experiments=train_experiment,
    time_window=time_window
)
test = extract_dataset_slice(
    dataset_path="dataset",
    substances=selected_CLASSES,
    features=selected_FEATURE,
    experiments=test_experiment,
    time_window=time_window
)
print(train)
print(test)

# CONSIGLIO PRATICO se la fase di loading del dataset richiede troppo tempo
# potreste valutare di salvare il dataset in un file /(spesso si usano i file.pkl)
# e salvare train e test una volta per tutte. Poi se volete provare altri dataset,
# bastera sovrascrivere o rinominare il file



#%%   separiamo i dati dalle lables
#
# tutte le libreire più note lavorano con variabili numeriche
# in formato numpy, anche i dati "categorici" come le classi (che soino semplicemente una lista di stringhe)


x_train = train.drop(columns=["CLASS"]).to_numpy()
y_train = train[["CLASS"]].to_numpy()
x_test = test.drop(columns=["CLASS"]).to_numpy()
y_test = test[["CLASS"]].to_numpy()

#%%  one hot encoding
# non è necessaria sempre, spesso possiamo dare dati direttamente in formato categorico
# ma cosi facendo, ci assicuriamo che venga usata questa codifica
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(y_train)

y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

# consiglio sempre di fare file separati dove implementare set di funzioni
# non è vera e proprio ingegneria de software ma aiuta ad acere il codice
# un po' ordinato
from chemical_brother.Classifier_function import train_mlp

n_of_inputs = x_train.shape[1]  # number of features
n_of_outputs = y_train.shape[1] # number of classes i.e. substances
# definisco il primo e il secondo hiden layer della rete neurale
# il livello di output è ovvio che abbia dimenzione pari a n_of_outputs
# e non serve specificarlo
hidden_layer_sizes = [n_of_inputs, 2*n_of_outputs]

# MLP sta per multy layers perceptrons, la rete neurale più comune
# Premendo ctrl/cmd e clikkando sulla funzione dovreste poter aprire
# il codice sorgente. Li potete modificare altri iperparamentri della rete
Neural_network_model = train_mlp(x_train, y_train,
                hidden_layer_sizes,
                learning_rate=0.0001)

# nessuno vi vieta di modificare la funzioe per passarli in ingresso
# invece che modificarli da li, cosi che potete usare lo stesso codice
# per provare vari setting, senza star sempre a modificare il codice sorgente


# NONNA NU MOLLA', CI SIAMO QUASI

#     I N F E R E N Z A
y_pred = Neural_network_model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# VEDIAMO SE FUNZIONA
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test_labels, y_pred_labels))

# l accuratezza globale non è una misura molto precisa, meglio la CM
from chemical_brother.Classifier_function import plot_confusion_matrix
plot_confusion_matrix(Neural_network_model, enc, x_test, y_test)

# e qui finisce il codice didattico. Ora non ci rimane che giocare col codice
# facendo qualche prova per verificare e fissare i concetti fatti durante il corso
#
# per esempio, vediamo che succede in caso di normalizzazione

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# scaler_test = StandardScaler()   assolutamente no
x_train = scaler.fit_transform(x_train)
# x_test = scaler_test.fit_transform(x_test) assolutamente no
x_test = scaler.transform(x_test)

Neural_network_model=train_mlp(x_train, y_train, hidden_layer_sizes,learning_rate=0.0001)

y_pred = Neural_network_model.predict(scaler.transform(x_test))
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
print("Test Accuracy scaled:", accuracy_score(y_test_labels, y_pred_labels))

plot_confusion_matrix(Neural_network_model, enc, x_test, y_test)
