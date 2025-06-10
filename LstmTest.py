"""
Testowanie modelu.
"""

import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
matplotlib.use('TkAgg')
from LstmModelClass import MyLSTMModel


def test_model(model, MoveData_test, MoveType_test, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    MoveData_tensor = torch.tensor(MoveData_test, dtype=torch.float32).to(device)
    MoveType_tensor = torch.tensor(MoveType_test, dtype=torch.long).to(device)

    dataset = TensorDataset(MoveData_tensor, MoveType_tensor)
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
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("---Quality of classification (avg from 2 classes)---")
    print(f'Accuracy on test set: {acc * 100:.2f}%')
    print(f'Precision:             {precision * 100:.2f}%')
    print(f'Recall:                {recall * 100:.2f}%')
    print(f'F1 Score:              {f1 * 100:.2f}%')

    labels = ['Walk', 'Run']

    # Macierz pomyłek i wizualizacja
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return acc


def test_from_pkl(pkl_path, model_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    MoveData_test = data['MoveData']
    MoveType_test = data['MoveType']

    input_size = MoveData_test.shape[2]
    hidden_size = 64
    num_layers = 2
    num_classes = 2

    model = MyLSTMModel(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return test_model(model, MoveData_test, MoveType_test)


if __name__ == "__main__":
    # Przykładowe uruchomienie bezpośrednie (opcjonalne)
    test_from_pkl('Data/testData.pkl', "Data/model.pth")
