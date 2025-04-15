import os
import yaml
import torch

from train import main as train_main
from predict import main as predict_main

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # === 路徑檢查 ===
    model_path = config['path']['save_path']
    model_exists = os.path.exists(model_path)

    if not model_exists:
        print("📦 No trained model found. Starting training...")
        train_main()
    else:
        print("✅ Found existing trained model, skip training.")

    print("\n🔍 Start prediction...")
    predict_main()

    print("\n🎉 Pipeline complete! Results saved as pred.json and pred.csv.")

if __name__ == '__main__':
    main()
