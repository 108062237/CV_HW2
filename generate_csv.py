import json
import pandas as pd
from collections import defaultdict

# category_id: 1~10 → 數字: 0~9
CATEGORY_TO_DIGIT = {
    1: '0',
    2: '1',
    3: '2',
    4: '3',
    5: '4',
    6: '5',
    7: '6',
    8: '7',
    9: '8',
    10: '9',
}

def load_predictions(pred_json_path):
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
    return preds

def group_predictions_by_image(preds, score_threshold=0.5):
    grouped = defaultdict(list)
    for pred in preds:
        if pred['score'] >= score_threshold:
            image_id = pred['image_id']
            x = pred['bbox'][0]  # x_min for sorting
            label = CATEGORY_TO_DIGIT.get(pred['category_id'], '?')  # fallback: ?
            grouped[image_id].append((x, label))
    return grouped

def generate_pred_label(grouped, all_image_ids):
    rows = []
    for image_id in all_image_ids:
        digits = grouped.get(image_id, [])
        if not digits:
            rows.append({'image_id': image_id, 'pred_label': -1})
        else:
            # Sort digits by x position
            sorted_digits = sorted(digits, key=lambda x: x[0])
            number_str = ''.join(d[1] for d in sorted_digits)
            rows.append({'image_id': image_id, 'pred_label': number_str})
    return rows

def get_all_image_ids(test_json_path):
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    return [img['id'] for img in data['images']]

def main():
    pred_json_path = 'pred.json'
    test_json_path = 'data/test.json'
    output_csv_path = 'pred.csv'
    threshold = 0.5

    preds = load_predictions(pred_json_path)
    all_image_ids = get_all_image_ids(test_json_path)
    grouped_preds = group_predictions_by_image(preds, score_threshold=threshold)
    result_rows = generate_pred_label(grouped_preds, all_image_ids)

    df = pd.DataFrame(result_rows)
    df.to_csv(output_csv_path, index=False)
    print(f"📁 Saved corrected pred.csv to {output_csv_path}")

if __name__ == '__main__':
    main()
