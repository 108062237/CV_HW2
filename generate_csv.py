import json
import pandas as pd
from collections import defaultdict

CATEGORY_TO_DIGIT = {
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
    6: '5', 7: '6', 8: '7', 9: '8', 10: '9'
}

def load_predictions(pred_json_path):
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
    return preds

def group_and_filter(preds, threshold=0.5):
    grouped = defaultdict(list)
    for pred in preds:
        if pred['score'] >= threshold:
            image_id = pred['image_id']
            x_min, y_min, w, h = pred['bbox']
            x_center = x_min + w / 2
            grouped[image_id].append({
                'x_center': x_center,
                'width': w,
                'score': pred['score'],
                'digit': CATEGORY_TO_DIGIT.get(pred['category_id'], '?')
            })

    # 過濾重疊數字（保留高信心）
    final_grouped = {}
    for image_id, digits in grouped.items():
        digits.sort(key=lambda d: d['x_center'])
        filtered = []
        for d in digits:
            if not filtered:
                filtered.append(d)
            else:
                prev = filtered[-1]
                dist = abs(d['x_center'] - prev['x_center'])
                if dist < (prev['width'] * 0.6):
                    if d['score'] > prev['score']:
                        filtered[-1] = d
                else:
                    filtered.append(d)
        final_grouped[image_id] = filtered
    return final_grouped

def get_all_image_ids(test_json_path):
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    return [img['id'] for img in data['images']]

def generate_pred_csv(pred_json_path, test_json_path, output_csv='pred.csv'):
    preds = load_predictions(pred_json_path)
    all_ids = get_all_image_ids(test_json_path)
    grouped = group_and_filter(preds)

    result_rows = []
    for img_id in all_ids:
        if img_id not in grouped or not grouped[img_id]:
            result_rows.append({'image_id': img_id, 'pred_label': -1})
        else:
            digits = [d['digit'] for d in grouped[img_id]]
            result_rows.append({'image_id': img_id, 'pred_label': ''.join(digits)})

    df = pd.DataFrame(result_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved improved Task 2 result to {output_csv}")

if __name__ == '__main__':
    generate_pred_csv('valid_pred.json', 'data/valid.json', output_csv='valid_pred.csv')
