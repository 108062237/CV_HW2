# HW2 - Digit Detection and Recognition

This is the homework 2 project for the course "Visual Recognition using Deep Learning (Spring 2025)".

The goal is to detect digits in images (Task 1) and predict the full digit sequence (Task 2).

---

## 📌 Tasks

- **Task 1**: Detect each digit and output bounding boxes → `pred.json`
- **Task 2**: Convert detection results into full numbers → `pred.csv`

---

## 📂 Files

- `train.py` – train Faster R-CNN
- `predict.py` – generate `pred.json`
- `generate_csv.py` – generate `pred.csv` from predictions
- `config.yaml` – configuration file
- `model.py`, `dataset.py` – model and dataset definitions

---

## 🚀 How to Run

Train the model:
```bash
python train.py
```

Make predictions:
```bash
python predict.py
```

Generate `pred.csv`:
```bash
python generate_csv.py
```


