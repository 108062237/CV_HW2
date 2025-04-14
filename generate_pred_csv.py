import json
import csv
from collections import defaultdict

def category_id_to_digit(category_id):
    return str(category_id - 1)  # 1 → "0", 10 → "9"

def generate_pred_csv(pred_json_path, image_list_json_path, output_csv_path="output/pred.csv", score_thresh=0.5):
    with open(pred_json_path, "r") as f:
        predictions = json.load(f)

    # ✅ 讀取所有圖片 id（補齊用）
    with open(image_list_json_path, "r") as f:
        image_data = json.load(f)
    all_image_ids = sorted([img["id"] for img in image_data["images"]])

    # ✅ group by image_id
    pred_dict = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= score_thresh:
            pred_dict[pred["image_id"]].append(pred)

    # ✅ 組合每張圖的數字
    results = []
    for image_id in all_image_ids:
        preds = pred_dict.get(image_id, [])
        if len(preds) == 0:
            results.append({"image_id": image_id, "pred_label": -1})
            continue

        preds = sorted(preds, key=lambda p: p["bbox"][0])
        digits = [category_id_to_digit(p["category_id"]) for p in preds]
        number_str = ''.join(digits)
        results.append({"image_id": image_id, "pred_label": number_str})

    with open(output_csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "pred_label"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Saved pred.csv to {output_csv_path}, total images: {len(results)}")

if __name__ == "__main__":
    generate_pred_csv(
        pred_json_path="output/pred.json",
        image_list_json_path="data/test.json",  # ← 用你的 test.json
        output_csv_path="output/pred.csv"
    )
