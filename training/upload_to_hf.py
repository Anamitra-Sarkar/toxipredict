import os
import json
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, upload_file

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "Arko007/toxipredict-gnn-models"


def upload_model(state_dict: dict, config: dict, calibration_data: dict = None):
    api = HfApi(token=HF_TOKEN)

    try:
        create_repo(REPO_ID, private=False, exist_ok=True, token=HF_TOKEN)
        print(f"Repository {REPO_ID} ready")
    except Exception as e:
        print(f"Repo error: {e}")

    os.makedirs("upload", exist_ok=True)

    save_file(state_dict, "upload/model.safetensors")
    print("Model weights saved to upload/model.safetensors")

    with open("upload/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Config saved to upload/model_config.json")

    if calibration_data:
        with open("upload/conformal_calibration.json", "w") as f:
            json.dump(calibration_data, f, indent=2)
        print("Calibration data saved")

    api.upload_file(
        path_or_fileobj="upload/model.safetensors",
        path_in_repo="model.safetensors",
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print("Uploaded model.safetensors")

    api.upload_file(
        path_or_fileobj="upload/model_config.json",
        path_in_repo="model_config.json",
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print("Uploaded model_config.json")

    if calibration_data:
        api.upload_file(
            path_or_fileobj="upload/conformal_calibration.json",
            path_in_repo="conformal_calibration.json",
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )

    print(f"\nModel uploaded to https://huggingface.co/{REPO_ID}")


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "checkpoints/model_final.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded checkpoint from {checkpoint_path}")

    from config import NODE_DIM, EDGE_DIM, HIDDEN_DIM, NUM_TASKS, DROPOUT, TASK_NAMES

    config = {
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_tasks": NUM_TASKS,
        "dropout": DROPOUT,
        "task_names": TASK_NAMES,
        "architecture": "MultiTaskGNN",
        "version": "1.0.0",
    }

    if os.path.exists("checkpoints/training_results.json"):
        with open("checkpoints/training_results.json") as f:
            config["training_results"] = json.load(f)

    upload_model(state_dict, config)


if __name__ == "__main__":
    main()
