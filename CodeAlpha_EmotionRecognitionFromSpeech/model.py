import torch.nn as nn


class EmotionCNNBiLSTM(nn.Module):
    def __init__(self, n_classes, n_mfcc=40):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),
        )
        self.lstm = nn.LSTM(128, 96, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.35), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(x.mean(dim=1))
