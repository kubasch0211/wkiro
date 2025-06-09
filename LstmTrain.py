"""
Trening modelu LSTM.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score
from LstmModelClass import MyLSTMModel

# HYPERPARAMETRY
WINDOW_SIZE = 100
NUM_FEATURES = 72  # 8 markerów * 3 (x,y,z)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def train_model(MoveData, MoveType):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Konwersja numpy na torch.Tensor
    MoveData_tensor = torch.tensor(MoveData, dtype=torch.float32)
    y_tensor = torch.tensor(MoveType, dtype=torch.long)

    dataset = TensorDataset(MoveData_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MyLSTMModel(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

            loop.set_postfix(loss=f"{total_loss / (loop.n + 1):.4f}")

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)

        train_losses.append(avg_loss)
        train_accuracies.append(acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")

    # Wykresy
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Trening zakończony")
    return model


def train_from_pkl(pkl_path, output_model_path):
    import pickle

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    MoveData = data['MoveData']
    MoveType = data['MoveType']

    model = train_model(MoveData, MoveType)
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
