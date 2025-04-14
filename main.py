import yaml
from models.faster_rcnn import train_model

if __name__ == "__main__":
    with open("configs/fasterrcnn.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_model(config)
