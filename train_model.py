import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # ✅ nowość

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_model(data_file='data.pkl', model_file='model.pt', epochs=10, batch_size=32):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[2]
    model = LSTMClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for X_batch, y_batch in tqdm(loader, desc=f"Training", leave=False):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_file)
    print(f"Model zapisany do {model_file}")
