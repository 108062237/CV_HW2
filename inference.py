import os
import json
import torch
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataloader.digit_dataset import DigitTestDataset  # 改為 test 專用 Dataset
from utils.get_transforms import get_transform

def get_model(num_classes, weight_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"📦 Loading model weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference(model, dataloader, device, output_path, score_thresh=0.5):
    results = []
    for images, targets in tqdm(dataloader, desc="Inferencing"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"])
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_thresh:
                    continue  # ← 過濾低信心預測
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                result = {
                    "image_id": image_id,
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "score": float(score),
                    "category_id": int(label)
                }
                results.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"✅ Saved prediction results to {output_path}")

def main():
    import yaml
    with open("configs/fasterrcnn.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- 選擇推論資料集（val 或 test） ----------
    use_test = True  # ← 改這裡即可切換
    if use_test:
        json_path = "data/test.json"
        image_dir = "data/test"
        transforms = get_transform("test")
        from dataloader.digit_dataset import DigitTestDataset as Dataset
    else:
        json_path = config["data"]["val_json"]
        image_dir = config["data"]["val_dir"]
        transforms = get_transform("val")
        from dataloader.digit_dataset import DigitDataset as Dataset

    dataset = Dataset(json_path=json_path, image_dir=image_dir, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(
        num_classes=config["model"]["num_classes"],
        weight_path=config["output"]["model_path"],
        device=device
    )

    run_inference(model, dataloader, device, output_path="output/pred.json", score_thresh=0.5)

if __name__ == "__main__":
    main()
