# 🪖 Helmet & License Plate Detection — AI Model Comparison

### YOLOv8 vs Faster R-CNN vs SSD300 | Tested on 42 Real Images

---

## 📌 Project Overview

This project answers one key question:

> **Which AI model performs best at detecting helmets and license plates in real-world images?**

We evaluate **three pre-trained object detection models** on the **same 42 test images**, measuring their:

* Accuracy (Precision, Recall, F1, mAP@0.5)
* Confidence
* Speed (Latency & FPS)

All results are visualized through charts and summarized in tables.

---

## 🎯 Objectives

* Compare 3 models under identical conditions
* Evaluate performance for a real-world traffic safety use case
* Analyze trade-offs between **accuracy vs speed**

---

## ⚠️ Important Note (Fairness of Comparison)


* **YOLOv8** is **task-specific** (trained on helmets & license plates)
* **Faster R-CNN** and **SSD300** are **general-purpose models** trained on the COCO dataset

➡️ Therefore:

* YOLO has **task knowledge advantage**
* The other models **lack helmet/plate awareness**

---

## 🧠 Models Used

| Model              | Source      | Training Data          | Classes                          |
| ------------------ | ----------- | ---------------------- | -------------------------------- |
| YOLOv8 (`best.pt`) | GitHub      | Helmet + Plate dataset | Plate, WithHelmet, WithoutHelmet |
| Faster R-CNN       | torchvision | COCO (80 classes)      | Person, bicycle, car, motorcycle |
| SSD300             | torchvision | COCO (80 classes)      | Person, bicycle, car, motorcycle |

---

## 🗂️ Dataset

Source: Roboflow Universe
Format: COCO (with bounding box annotations)

| Split     |  Images |
| --------- | ------: |
| Train     |     681 |
| Valid     |      64 |
| Test      |      42 |
| **Total** | **787** |

---

## ⚙️ Methodology

### 1. Setup

* Installed dependencies via `pip`
* Ran on **CPU**

---

### 2. Data Loading

* Dataset downloaded via Roboflow API
* Ground truth boxes loaded from COCO annotations

---

### 3. Test Selection

* Used all **42 test images**
* `random.seed(42)` ensures reproducibility

---

### 4. Model Inference

* All models use:

  * Confidence threshold = **0.4**
* COCO models filtered to:

  * Person, bicycle, car, motorcycle

---

### 5. Evaluation (IoU)

A detection is considered **correct** if:

* IoU ≥ **0.5**

---

### 6. Metrics

* **Precision**
* **Recall**
* **F1 Score**
* **mAP@0.5**
* **Average Confidence**
* **Inference Time (ms)**
* **FPS**

---

## 📊 Results

```
Model          Avg Det   Avg Conf   Time(ms)   FPS   Precision   Recall   F1     mAP@0.5
-----------------------------------------------------------------------------------------
YOLOv8            0.74      0.479      218.8   4.6       0.571    0.117   0.188     0.143
Faster R-CNN     13.45      0.794     1883.4   0.5       0.258    0.219   0.208     0.202
SSD300            5.55      0.710      395.8   2.5       0.394    0.166   0.194     0.157
```

---

## 📈 Key Findings

### 🟢 YOLOv8

* Highest **Precision (0.571)**
* Very low **Recall (0.117)**
* Fastest model (**4.6 FPS**)
* Conservative — detects only when highly confident

---

### 🔴 Faster R-CNN

* Best **overall accuracy**

  * Highest **mAP (0.202)**
  * Highest **F1 (0.208)**
* Detects many objects (high recall)
* Extremely slow (**0.5 FPS**)

---

### 🟠 SSD300

* Balanced performance
* Moderate speed (**2.5 FPS**)
* Middle accuracy across all metrics

---

## 📊 Observations

* All models show **low performance overall**
* Reason:

  * Dataset is **task-specific**
  * COCO models lack relevant classes
* YOLO performs better in **precision**
* Faster R-CNN performs better in **recall**

---

## 📁 Project Structure

```
AI-PROJECT/
│
├── annotated_output/                  ← Output images with drawn annotations
│
├── Comparison_Results/                ← All results from the comparison notebook
│   ├── test_100/                      ← The 42 test images used in the comparison
│   ├── comparison_charts.png          ← Bar charts comparing all 7 metrics
│   ├── f1_per_image.png               ← F1 score line chart across all 42 images
│   ├── per_image_results.csv          ← Full results for every single image (42 rows)
│   ├── summary.csv                    ← Average results per model
│   └── vis_*.png                      ← Side-by-side visual detection samples
│
├── models/
│   └── best.pt                        ← YOLOv8 weights downloaded from GitHub
│
├── Project_Results1/                  ← Additional results folder
├── results/                           ← General results folder
│
├── src/
│   ├── helmet_plate_dataset/          ← Full dataset downloaded from Roboflow
│   │   ├── train/                     ← 681 images + labels (not used in comparison)
│   │   ├── valid/                     ← 64 images + labels (not used in comparison)
│   │   └── test/                      ← 42 images + _annotations.coco.json ← USED
│   ├── test_100/                      ← Local copy of test images
│   ├── AI_comparison_100images.ipynb  ← Main comparison notebook ← RUN THIS
│   └── AI_first.ipynb                 ← First/exploratory notebook
│
├── test_video/                        ← Test video files
├── venv/                              ← Python virtual environment (do not edit)
├── .gitattributes                     ← Git settings file
└── requirements.txt                   ← Python libraries to install
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

1. Add your Roboflow API key
2. Place `best.pt` inside `models/`
3. Update paths in notebook
4. Run:

```
AI_comparison_100images.ipynb
```

---

## 📦 Requirements

* torch
* torchvision
* ultralytics
* roboflow
* opencv-python-headless
* numpy
* pandas
* matplotlib

---

## 📊 Outputs

* `summary.csv` → average results
* `per_image_results.csv` → per-image metrics
* `comparison_charts.png` → bar charts
* `f1_per_image.png` → F1 trend
* `vis_*.png` → visual comparisons

---

## 📎 Dataset Source

Roboflow Universe — Helmet & License Plate Detection
