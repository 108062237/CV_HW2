import os
import json
import yaml
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DigitDetectionDataset
from model import get_faster_rcnn_model


def collate_fn(batch):
    return tuple(zip(*batch))


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()
    results = []

    for images, targets in tqdm(data_loader, desc="Predicting"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = int(target['image_id'].item())
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                x_min, y_min, x_max, y_max = box.tolist()
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                result = {
                    'image_id': image_id,
                    'bbox': bbox,
                    'score': float(score),
                    'category_id': int(label)
                }
                results.append(result)

    return results


def main():
    config = load_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # === Dataset
    test_dataset = DigitDetectionDataset(
        json_path='data/test.json',
        image_dir='data/test',
        transforms=ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # === Model
    model = get_faster_rcnn_model(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['name'],
        pretrained=False
    ).to(device)
    model.load_state_dict(torch.load(config['path']['save_path'], map_location=device))
    print("✅ Model loaded.")

    # === Predict & Save
    pred_results = predict(model, test_loader, device)

    with open('pred.json', 'w') as f:
        json.dump(pred_results, f)
    print("📁 Saved prediction to pred.json")


if __name__ == '__main__':
    main()
