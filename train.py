import os
import yaml
import torch
import time
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataset import DigitDetectionDataset
from model import get_faster_rcnn_model
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_boxes = 0
    correct_boxes = 0

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        # optional: custom eval logic
        # here we just print the number of predicted boxes
        total_boxes += sum(len(o['boxes']) for o in outputs)
        correct_boxes += sum(len(t['boxes']) for t in targets)  # not real accuracy!

    print(f"[Eval] Total predicted boxes: {total_boxes}")
    print(f"[Eval] Total GT boxes: {correct_boxes}")
    # You can replace this with real mAP calculation if needed
    return

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20):
    model.train()
    running_loss = 0.0

    for idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()


    avg_loss = running_loss / len(data_loader)
    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")


def main():
    config = load_config()

    # === Device ===
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # === Dataset ===
    transform = ToTensor()
    train_dataset = DigitDetectionDataset(
        json_path=config['path']['train_json'],
        image_dir=config['path']['train_images'],
        transforms=transform
    )
    val_dataset = DigitDetectionDataset(
        json_path=config['path']['val_json'],
        image_dir=config['path']['val_images'],
        transforms=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # === Model ===
    model = get_faster_rcnn_model(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['name'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # === Optimizer & Scheduler ===
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['train']['lr'],
                                  weight_decay=config['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['train']['step_size'],
        gamma=config['train']['gamma']
    )

    # === Training ===
    best_loss = float('inf')
    for epoch in range(config['train']['epochs']):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        # === Save model based on training loss (or mAP if you add mAP calc)
        save_dir = os.path.dirname(config['path']['save_path'])
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), config['path']['save_path'])
        print(f"✅ Model checkpoint saved at {config['path']['save_path']}")

        # === Optional evaluation
        evaluate(model, val_loader, device=device)

if __name__ == '__main__':
    main()
