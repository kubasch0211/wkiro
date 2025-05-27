import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from train_model import LSTMClassifier

def test_model(data_file='data.pkl', model_file='model.pt'):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = data['y_test']

    input_size = X_test.shape[2]
    model = LSTMClassifier(input_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    y_pred = predicted.numpy()

    # raport
    print("=== Raport klasyfikacji ===")
    print(classification_report(y_test, y_pred, target_names=["Walk", "Run"]))

    # macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Walk", "Run"], yticklabels=["Walk", "Run"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
