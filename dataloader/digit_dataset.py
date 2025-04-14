import os
import json
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class DigitDataset(Dataset):
    def __init__(self, json_path, image_dir, transforms=None):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transforms = transforms

        self.id_to_filename = {img['id']: img['file_name'] for img in self.data['images']}
        self.annotations = self.data['annotations']

        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            self.img_id_to_anns.setdefault(img_id, []).append(ann)

        self.category_id_to_digit = {
            cat["id"]: cat["name"] for cat in self.data.get("categories", [])
        }

        self.ids = list(self.id_to_filename.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, self.id_to_filename[img_id])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self.img_id_to_anns.get(img_id, [])
        h, w = img.shape[:2]
        boxes = []
        labels = []

        
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            x1 = max(0, min(x, w - 1))
            y1 = max(0, min(y, h - 1))
            x2 = max(0, min(x + bw, w - 1))
            y2 = max(0, min(y + bh, h - 1))
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, category_ids=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["category_ids"]
        else:
            from torchvision.transforms.functional import to_tensor
            img = to_tensor(img)

        return img, target
    
class DigitTestDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_dir, transforms=None):
        with open(json_path) as f:
            data = json.load(f)
        self.image_dir = image_dir
        self.transforms = transforms
        self.images = data["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        entry = self.images[idx]
        img_path = os.path.join(self.image_dir, entry["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]
        return img, {"image_id": entry["id"]}


# ========= 測試區塊 =========
if __name__ == "__main__":
    dataset = DigitDataset(
        json_path="data/train.json",
        image_dir="data/train"
    )

    print(f"Total samples: {len(dataset)}")

    for i in range(5):
        image, target = dataset[i]
        img = image.copy()

        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = map(int, box.tolist())

            true_digit = dataset.category_id_to_digit[label.item()]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, true_digit, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        plt.figure(figsize=(5, 3))
        plt.imshow(img)
        plt.title(f"Image ID: {target['image_id'].item()}")
        plt.axis('off')
        plt.show()
