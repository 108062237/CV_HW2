import os
import json
import torch
import yaml
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model import get_faster_rcnn_model
from dataset import DigitDetectionDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def collate_fn(batch):
    return tuple(zip(*batch))

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def predict_and_show(model, data_loader, device, output_json='pred.json', show_limit=5):
    model.eval()
    results = []
    shown = 0

    for images, targets in tqdm(data_loader, desc="Predicting"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for img_tensor, output, target in zip(images, outputs, targets):
            image_id = int(target['image_id'].item())

            # 儲存預測資料到 pred.json 格式
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                x_min, y_min, x_max, y_max = box.tolist()
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                results.append({
                    'image_id': image_id,
                    'bbox': bbox,
                    'score': float(score),
                    'category_id': int(label)
                })

            # 僅顯示前 N 張圖
            if shown < show_limit:
                img_np = img_tensor.mul(255).permute(1, 2, 0).byte().cpu().numpy().copy()

                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    x_min, y_min, x_max, y_max = box.int().tolist()
                    cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img_np, f"{label.item() - 1}", (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                os.makedirs("output_vis", exist_ok=True)
                save_path = f"output_vis/image_{image_id}.jpg"
                cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                print(f"🖼️ Saved preview: {save_path}")
                shown += 1

    with open(output_json, 'w') as f:
        json.dump(results, f)
    print(f"✅ Saved prediction to {output_json}")

def evaluate_map(gt_json='data/valid.json', pred_json='pred.json'):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@IoU=0.5:0.95

def main():
    config = load_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    val_dataset = DigitDetectionDataset(
        config['path']['val_json'],
        config['path']['val_images'],
        transforms=ToTensor()
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_faster_rcnn_model(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['name'],
        pretrained=False
    ).to(device)

    model.load_state_dict(torch.load(config['path']['save_path'], map_location=device))
    print("✅ Loaded model.")

    predict_and_show(model, val_loader, device, output_json='valid_pred.json', show_limit=0)

    map_score = evaluate_map(gt_json=config['path']['val_json'], pred_json='valid_pred.json')
    print(f"\n🎯 Validation mAP@[0.5:0.95] = {map_score:.4f}")

def evaluate_map(gt_json='data/valid.json', pred_json='pred.json'):
    # 1. 載入 ground truth
    coco_gt = COCO(gt_json)

    # 2. 載入預測結果
    coco_dt = coco_gt.loadRes(pred_json)

    # 3. 初始化 COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # 4. 執行評估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 輸出 mAP
    return coco_eval.stats[0]  # mAP@[IoU=0.5:0.95]

if __name__ == '__main__':
    main()
