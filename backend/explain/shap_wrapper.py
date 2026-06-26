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
        with torch.no_grad():
            for coalition in binary_coalitions:
                x_perturbed = self.data_obj.x.clone().to(self.device)
                mask_tensor = torch.tensor(
                    coalition, dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                x_perturbed = x_perturbed * mask_tensor

                logits = self.model(
                    x_perturbed,
                    self.data_obj.edge_index.to(self.device),
                    self.data_obj.edge_attr.to(self.device),
                    torch.zeros(x_perturbed.shape[0], dtype=torch.long, device=self.device),
                )
                probs = torch.sigmoid(logits)
                predictions.append(probs[0, self.target_task_idx].cpu().item())

        return np.array(predictions)
