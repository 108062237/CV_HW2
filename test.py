import os
import json
import cv2
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def generate_test_json(image_dir, save_path="data/123123.json"):
    image_list = sorted(os.listdir(image_dir), key=natural_sort_key)
    images = []
    id_counter = 1
    for filename in image_list:
        if not filename.lower().endswith(".png"):
            continue
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 跳過無法讀取的圖片：{filename}")
            continue
        h, w = img.shape[:2]
        images.append({
            "id": id_counter,
            "file_name": filename,
            "height": h,
            "width": w
        })
        id_counter += 1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"images": images}, f, indent=2)
    print(f"✅ 已儲存 test.json（共 {len(images)} 張圖片）到：{save_path}")

if __name__ == "__main__":
    generate_test_json("data/valid")
