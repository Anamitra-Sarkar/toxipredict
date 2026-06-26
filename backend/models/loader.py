import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from config import HF_MODEL_REPO, HF_TOKEN, MODEL_CACHE_DIR, NODE_DIM, EDGE_DIM, HIDDEN_DIM, NUM_TASKS, DROPOUT
from .multitask_gnn import MultiTaskGNN


class ModelLoader:
    def __init__(self):
        self.model = None
        self.config = None

    def load_config(self) -> dict:
        if self.config is not None:
            return self.config
        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="model_config.json",
                token=HF_TOKEN or None,
                cache_dir=MODEL_CACHE_DIR,
            )
            with open(path) as f:
                self.config = json.load(f)
        except Exception:
            self.config = {
                "node_dim": NODE_DIM,
                "edge_dim": EDGE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_tasks": NUM_TASKS,
                "dropout": DROPOUT,
            }
        return self.config

    def load_model(self) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        config = self.load_config()

        model = MultiTaskGNN(
            in_channels=config.get("node_dim", NODE_DIM),
            edge_dim=config.get("edge_dim", EDGE_DIM),
            hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
            num_tasks=config.get("num_tasks", NUM_TASKS),
            dropout=config.get("dropout", DROPOUT),
        )

        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="model.safetensors",
                token=HF_TOKEN or None,
                cache_dir=MODEL_CACHE_DIR,
            )
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
        except Exception:
            pass

        model.eval()
        self.model = model
        return model

    def is_loaded(self) -> bool:
        return self.model is not None
