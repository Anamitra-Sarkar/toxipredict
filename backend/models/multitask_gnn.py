import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class MultiTaskGNN(nn.Module):
    def __init__(self, in_channels: int, edge_dim: int, hidden_dim: int,
                 num_tasks: int, dropout: float = 0.15):
        super(MultiTaskGNN, self).__init__()
        self.num_tasks = num_tasks

        self.conv1 = GATConv(in_channels, hidden_dim, heads=4, concat=True,
                             edge_dim=edge_dim, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)

        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(self.bn1(h))

        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(self.bn2(h))

        hg = global_mean_pool(h, batch)
        shared_out = self.fc_shared(hg)

        logits = torch.cat([head(shared_out) for head in self.heads], dim=1)
        return logits
