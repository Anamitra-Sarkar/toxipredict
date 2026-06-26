"""
Start Kaggle GPU Training for ToxiPredict.
Uploads our training script and runs it on Kaggle's GPU.
"""
import os
import json
import shutil
import kagglehub
from kagglehub import KaggleApiAdapter

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "kaggle_notebook")
KAGGLE_USERNAME = "anamitrasarkar007"
NOTEBOOK_SLUG = "toxipredict-training"


def deploy_to_kaggle():
    print("=" * 60)
    print("Deploying ToxiPredict training to Kaggle GPU")
    print("=" * 60)

    os.chdir(NOTEBOOK_DIR)

    print("\nInstalling kagglehub CLI...")
    try:
        from kagglehub.cli import push_kernel
        push_kernel(
            kernel_slug=NOTEBOOK_SLUG,
            owner_slug=KAGGLE_USERNAME,
            language="python",
            kernel_type="script",
            enable_gpu=True,
            enable_internet=True,
            is_private=False,
        )
        print(f"Notebook pushed: https://kaggle.com/{KAGGLE_USERNAME}/{NOTEBOOK_SLUG}")
    except Exception as e:
        print(f"CLI push failed: {e}")
        print("\nManual upload required:")
        print(f"1. Go to https://www.kaggle.com/{KAGGLE_USERNAME}/new-notebook")
        print(f"2. Upload the file: {NOTEBOOK_DIR}/toxipredict_train.py")
        print("3. Enable GPU accelerator")
        print("4. Add internet access")
        print("5. Set environment variable HF_TOKEN")
        print("6. Click Run")


def print_instructions():
    print("\n" + "=" * 60)
    print("MANUAL INSTRUCTIONS")
    print("=" * 60)
    print(f"""
1. The training script is ready at:
   {NOTEBOOK_DIR}/toxipredict_train.py

2. To run on Kaggle GPU:
   - Go to https://www.kaggle.com/{KAGGLE_USERNAME}/new-notebook
   - Click "File > Upload Notebook" and select the file above
     OR paste the content into a new notebook
   - In Settings panel:
     * Accelerator: GPU T4 x2 (or P100)
     * Internet: ON
     * Environment: Default (Python)
   - Add secret: HF_TOKEN = <your_hf_token>
   - Click "Run All" (will take ~2-3 hours for full training)

3. The model will automatically upload to:
   https://huggingface.co/Arko007/toxipredict-gnn-models

4. After training completes, the backend will auto-download
   the model from HuggingFace at startup.

5. Expected Results (from research plan):
   - Mean AUC-ROC >= 0.84 (Tox21, scaffold 5-fold CV)
   - Mean AUC-ROC >= 0.79 (MUV, scaffold 5-fold CV)
   - F1-Score >= 0.76
""")
    print("Training requirements:")
    print("  - GPU: T4 or better (Kaggle provides free T4 x2)")
    print("  - Memory: >= 8GB GPU RAM")
    print("  - Time: ~2-3 hours")
    print("  - Dataset downloads automatically from DeepChem")


if __name__ == "__main__":
    deploy_to_kaggle()
    print_instructions()
