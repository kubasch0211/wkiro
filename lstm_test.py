import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn


class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(MyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # dropout w LSTM, tylko gdy num_layers > 1
        )
        self.dropout = nn.Dropout(dropout)  # dodatkowy dropout
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # ostatni krok czasowy
        out = self.dropout(out)      # dropout przed warstwą fc
        out = self.fc(out)
        return out


def test_model(model, X_test, y_test, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f'Accuracy on test set: {acc * 100:.2f}%')

    # Macierz pomyłek i wizualizacja
    cm = confusion_matrix(all_labels, all_preds)
    labels = ['Walk', 'Run']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return acc


def test_from_pkl(pkl_path, model_path="model.pth"):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    X_test = data['X']
    y_test = data['y']

    input_size = X_test.shape[2]
    hidden_size = 64
    num_layers = 2
    num_classes = 2

    model = MyLSTMModel(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return test_model(model, X_test, y_test)


if __name__ == "__main__":
    # Przykładowe uruchomienie bezpośrednie (opcjonalne)
    test_from_pkl('Data/testData.pkl', "model.pth")
