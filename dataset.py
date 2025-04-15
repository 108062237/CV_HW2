import os
import json
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class DigitDetectionDataset(Dataset):
    def __init__(self, json_path, image_dir, transforms=None):
        with open(json_path) as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.transforms = transforms

        self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        self.ids = list(self.id_to_filename.keys())

        # Optional annotations
        self.img_id_to_anns = {}
        for ann in data.get('annotations', []):
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.has_annotation = len(self.img_id_to_anns) > 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, self.id_to_filename[img_id])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)

        target = {'image_id': torch.tensor([img_id])}

        if self.has_annotation:
            anns = self.img_id_to_anns.get(img_id, [])
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)

        return img, target
