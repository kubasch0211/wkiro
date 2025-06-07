import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# HYPERPARAMETRY
WINDOW_SIZE = 100
NUM_FEATURES = 24  # 8 markerów * 3 (x,y,z)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(MyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # ostatni krok czasowy
        out = self.dropout(out)
        out = self.fc(out)
        return out

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

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{total_loss / (loop.n + 1):.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

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
