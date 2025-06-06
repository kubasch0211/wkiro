import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# HYPERPARAMS
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

def train_model(X, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Konwersja numpy na torch.Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MyLSTMModel(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # Używamy tqdm, by pokazać pasek postępu dla batchy
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # aktualizacja paska postępu: wyświetlamy aktualny średni loss
            loop.set_postfix(loss=f"{total_loss / (loop.n + 1):.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("Trening zakończony")
    return model

# PRZYKŁADOWE WYWOŁANIE:
if __name__ == "__main__":
    import pickle
    with open('C:/WKIRO/wkiro/Data/trainData.pkl', 'rb') as f:
        data = pickle.load(f)
    X = data['X']  # kształt: (num_samples, WINDOW_SIZE, num_features)
    y = data['y']

    model = train_model(X, y)
    torch.save(model.state_dict(), "model.pth")