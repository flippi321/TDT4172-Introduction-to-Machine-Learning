import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1), 
        )

    def forward(self, x):
        return self.net(x)

