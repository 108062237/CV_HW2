import os
import json
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.digit_dataset import DigitDataset
from utils.get_transforms import get_transform
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_model(config):
    name = config['model']['name']
    num_classes = config['model']['num_classes']

    if name == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise NotImplementedError(f"Model '{name}' is not supported yet.")
    return model

@torch.no_grad()
def evaluate_map(model, val_loader, device, ann_file, output_json="output/temp_val_pred.json"):
    model.eval()
    coco_results = []

    for images, targets in tqdm(val_loader, desc="Evaluating (for mAP)"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"])
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                coco_results.append({
                    "image_id": image_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "category_id": int(label)
                })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_results, f)

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(output_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # AP@[IoU=0.50:0.95]

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DigitDataset(config['data']['train_json'], config['data']['train_dir'], transforms=get_transform("train"))
    val_dataset = DigitDataset(config['data']['val_json'], config['data']['val_dir'], transforms=get_transform("val"))

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)

    model = get_model(config)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    best_val_loss = float("inf")
    loss_log = []

    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}]")
        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            loop.set_postfix(loss=losses.item())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_map(model, val_loader, device, config['data']['val_json'])

        print(f"Train Loss: {avg_train_loss:.4f} | Val mAP: {avg_val_loss:.4f}")
        loss_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_map": avg_val_loss
        })


        # 儲存最佳 checkpoint
        if avg_val_loss > best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
            torch.save(model.state_dict(), config['output']['model_path'])
            print(f"✅ Best model updated (val_mAP={best_val_loss:.4f})")

    # 畫圖與儲存 loss/mAP curve
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(loss_log)
    df.to_csv("output/loss_log.csv", index=False)

    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_map"], label="Val mAP")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / mAP")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig("output/loss_map_curve.png")
    print("📈 Saved training log and curve.")

    return model

