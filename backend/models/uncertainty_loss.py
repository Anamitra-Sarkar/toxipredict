import torch
import torch.nn as nn
import torch.nn.functional as F


class HomoscedasticUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks: int):
        super(HomoscedasticUncertaintyLoss, self).__init__()
        self.num_tasks = num_tasks
        self.s = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(self, logits, targets, mask):
        total_loss = 0.0
        task_losses = []
        task_weights = torch.exp(-self.s)
        valid_task_count = 0

        for t in range(self.num_tasks):
            task_logits = logits[:, t]
            task_targets = targets[:, t]
            task_mask = mask[:, t]

            valid_indices = torch.where(task_mask == 1)[0]
            if len(valid_indices) == 0:
                task_losses.append(torch.tensor(0.0, device=logits.device))
                continue

            v_logits = task_logits[valid_indices]
            v_targets = task_targets[valid_indices].float()

            bce_loss = F.binary_cross_entropy_with_logits(
                v_logits, v_targets, reduction='mean'
            )
            task_losses.append(bce_loss)

            weighted_loss = task_weights[t] * bce_loss + 0.5 * self.s[t]
            total_loss += weighted_loss
            valid_task_count += 1

        if valid_task_count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), task_losses, task_weights

        return total_loss, torch.stack(task_losses) if task_losses else torch.tensor([]), task_weights
