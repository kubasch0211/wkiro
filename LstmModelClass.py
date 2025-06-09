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
        self.dropout = nn.Dropout(dropout) # dodatkowy dropout
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]     # ostatni krok czasowy
        out = self.dropout(out) # dropout przed warstwą fc
        out = self.fc(out)
        return out