import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 25002
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = False
DROPOUT = 0.5


class LSTMBase(nn.Module):
    def __init__(self):
        super(LSTMBase, self).__init__()
        self.embedding = nn.Embedding(INPUT_DIM, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=N_LAYERS, bidirectional=BIDIRECTIONAL,
                            dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2, OUTPUT_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden.squeeze(0))


def lstm(pretrained=False, mode_path=None, device="cpu"):
    if pretrained:
        if "_@s" in mode_path:
            model = LSTMBase().to(device)
            model.load_state_dict(torch.load(mode_path, map_location=device))
        elif "_@m" in mode_path:
            model = torch.load(mode_path, map_location=device)
    else:
        model = LSTMBase().to(device)
    return model
