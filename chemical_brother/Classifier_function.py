from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

def train_mlp(x_train, y_train, hidden_layer_sizes, learning_rate=0.0001):

    model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes),
                          max_iter=230,
                          activation='relu',
                          solver='adam',
                          random_state=42,
                          verbose=True,
                          learning_rate_init=learning_rate)
    model.fit(x_train, y_train)

    return model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(model, encoder, X_test, y_test):
    """
    Plot the confusion matrix using the original class names.

    Parameters:
    model (MLPClassifier): Trained neural network model.
    encoder (OneHotEncoder): Fitted label encoder.
    X_test (numpy.ndarray): Test feature matrix.
    y_test (numpy.ndarray): One-hot encoded test labels.
    """
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    class_labels = encoder.categories_[0]
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()