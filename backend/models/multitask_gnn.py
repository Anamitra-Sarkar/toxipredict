import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class ResidualGATv2Layer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.15):
        super().__init__()
        self.conv = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False,
                              edge_dim=edge_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = F.relu(self.norm(h))
        return h + x


class MultiTaskGNN_ResGATv2_JK_VN(nn.Module):
    def __init__(self, in_channels: int, edge_dim: int, hidden_dim: int,
                 num_tasks: int, dropout: float = 0.15):
        super().__init__()
        self.num_tasks = num_tasks

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.convs = nn.ModuleList([
            ResidualGATv2Layer(hidden_dim, edge_dim, dropout)
            for _ in range(3)
        ])

        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        h = self.input_proj(x)

        layer_outputs = [h]
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            layer_outputs.append(h)

        jk_cat = torch.cat(layer_outputs, dim=-1)
        h_node = self.jk_proj(jk_cat)

        h_mean = global_mean_pool(h_node, batch)
        h_max = global_max_pool(h_node, batch)

        h_cat = torch.cat([h_mean, h_max], dim=-1)
        h_out = self.fc(h_cat)

        logits = torch.cat([head(h_out) for head in self.heads], dim=1)
        return logits
