import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.0
        ))
    
    elif mode == "val":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids']
        ))
    
    elif mode == "test":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])  # ❗️注意：無 bbox_params，因為 test 沒有標註
    
    else:
        raise ValueError(f"[get_transform] Invalid mode '{mode}'! Must be one of: 'train', 'val', 'test'")
