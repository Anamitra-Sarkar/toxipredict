import torch
import numpy as np


class GNNFeatureWrapper:
    def __init__(self, model, data_obj, target_task_idx: int, device: str):
        self.model = model.to(device)
        self.data_obj = data_obj
        self.target_task_idx = target_task_idx
        self.device = device
        self.model.eval()

    def __call__(self, binary_coalitions: np.ndarray) -> np.ndarray:
        predictions = []
        x = self.data_obj.x.to(self.device)
        edge_index = self.data_obj.edge_index.to(self.device)
        edge_attr = self.data_obj.edge_attr.to(self.device)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for coalition in binary_coalitions:
                mask = torch.tensor(coalition, dtype=torch.float32, device=self.device).unsqueeze(1)
                x_perturbed = x * mask
                logits = self.model(x_perturbed, edge_index, edge_attr, batch)
                prob = torch.sigmoid(logits[0, self.target_task_idx]).cpu().item()
                predictions.append(prob)

        return np.array(predictions)
