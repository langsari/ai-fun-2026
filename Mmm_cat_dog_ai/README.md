# 🐶🐱 Cat vs Dog AI Classification Project
## Deep Learning Model Comparison, Ensemble Evaluation, and Deployment

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

# 🇬🇧 English Version

## 1. Project Overview
This project builds an Artificial Intelligence (AI) system for **cat and dog image classification** using deep learning.

The project does **not** stop at training only one model.  
It includes:

- training **3 different models**
- comparing their performance
- evaluating an **ensemble method**
- deploying the trained AI for real image prediction

This makes the project suitable for both:
- **academic submission**
- **practical AI demonstration**

---

## 2. Main Files in This Project

### Core files
- `train.py`  
  Trains **SimpleCNN**, **MobileNetV2**, and **ResNet50**, then saves results and trained models.

- `evaluate_ensemble.py`  
  Loads the 3 trained models and evaluates the **ensemble average model** on the test set.

- `predict.py`  
  Runs prediction from the command line using the trained models.

- `app_gradio.py`  
  Launches a simple web app interface for AI prediction.

- `summary_results_formal.ipynb`  
  Notebook for formal analysis, comparison tables, graphs, and final conclusions.

- `requirements.txt`  
  Stores required Python packages.

---

## 3. Project Structure

```text
DOG_CAT_AI/
│
├── train.py
├── evaluate_ensemble.py
├── predict.py
├── app_gradio.py
├── summary_results_formal.ipynb
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── val/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── results/
│   ├── summary_results.json
│   ├── ensemble_results.json
│   ├── SimpleCNN_accuracy.png
│   ├── SimpleCNN_loss.png
│   ├── MobileNetV2_accuracy.png
│   ├── MobileNetV2_loss.png
│   ├── ResNet50_accuracy.png
│   ├── ResNet50_loss.png
│   ├── *_classification_report.json
│   ├── *_confusion_matrix.csv
│   └── ensemble_confusion_matrix.csv
│
└── saved_models/
    ├── SimpleCNN.keras
    ├── MobileNetV2.keras
    └── ResNet50.keras
```

---

## 4. Dataset Format
The dataset must be separated into **three subsets**:

- `train`
- `val`
- `test`

Each subset must contain **two folders**:

- `cats`
- `dogs`

Example:

```text
dataset/train/cats/
dataset/train/dogs/
dataset/val/cats/
dataset/val/dogs/
dataset/test/cats/
dataset/test/dogs/
```

Recommended split:
- Train = **70%**
- Validation = **15%**
- Test = **15%**

---

## 5. Installation

### Step 1: Create virtual environment
```bash
python -m venv venv
```

### Step 2: Activate virtual environment
Windows:
```bash
venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Full Workflow (Step by Step)

## Step 1 — Train the 3 individual models
Run:

```bash
python train.py
```

### What `train.py` does
This file:

1. loads dataset from:
   - `dataset/train`
   - `dataset/val`
   - `dataset/test`

2. trains the following models:
   - `SimpleCNN`
   - `MobileNetV2`
   - `ResNet50`

3. evaluates each model on the **test set**

4. saves:
   - trained models into `saved_models/`
   - graphs into `results/`
   - summary results into `results/summary_results.json`

### Expected output files after running `train.py`
In `saved_models/`:
- `SimpleCNN.keras`
- `MobileNetV2.keras`
- `ResNet50.keras`

In `results/`:
- `summary_results.json`
- `SimpleCNN_accuracy.png`
- `SimpleCNN_loss.png`
- `MobileNetV2_accuracy.png`
- `MobileNetV2_loss.png`
- `ResNet50_accuracy.png`
- `ResNet50_loss.png`
- `*_classification_report.json`
- `*_confusion_matrix.csv`

---

## Step 2 — Evaluate Ensemble model
Run:

```bash
python evaluate_ensemble.py
```

### What `evaluate_ensemble.py` does
This file:

1. loads the trained models from `saved_models/`
2. loads the **test dataset**
3. gets predictions from:
   - SimpleCNN
   - MobileNetV2
   - ResNet50
4. combines them into an **ensemble result**
5. saves ensemble evaluation files into `results/`

### Expected output files after running `evaluate_ensemble.py`
- `results/ensemble_results.json`
- `results/ensemble_classification_report.json`
- `results/ensemble_confusion_matrix.csv`

---

## Step 3 — Analyze results in notebook
Open:

```text
summary_results_formal.ipynb
```

Then click **Run All**.

### What the notebook shows
- project overview
- workflow explanation
- full comparison table
- model ranking
- accuracy comparison
- precision comparison
- recall comparison
- F1-score comparison
- training time comparison
- best single model vs ensemble
- final conclusion

---

## Step 4 — Predict a single image from command line
Run:

```bash
python predict.py your_image.jpg
```

### Example
```bash
python predict.py dataset/test/cats/cat.1.jpg
```

### Expected output
```text
Prediction: cats
Confidence: 0.9614
SimpleCNN: 0.1130
MobileNetV2: 0.0021
ResNet50: 0.0006
```

### Meaning
- `Prediction` = final predicted class
- `Confidence` = final confidence score
- the three model values show the probability from each individual model

---

## Step 5 — Run the AI web app
Run:

```bash
python app_gradio.py
```

After that, open:

```text
http://127.0.0.1:7860
```

### What the web app does
- lets the user upload an image
- predicts whether the image is a **cat** or a **dog**
- shows prediction confidence

This is the **deployment/demo part** of the project.

---

## 7. Model Summary

### Models used
1. **SimpleCNN**
   - baseline model
   - built from scratch
   - useful for comparison

2. **MobileNetV2**
   - lightweight pretrained model
   - good speed and strong performance

3. **ResNet50**
   - deeper pretrained model
   - best performance in this project

4. **Ensemble**
   - combines predictions from multiple models
   - evaluated to see whether combining models improves performance

---

## 8. Final Result Interpretation
Based on the project results:

- **ResNet50** achieved the best performance
- **MobileNetV2** also performed very well
- **SimpleCNN** had the lowest performance, but it is still important as a baseline
- **Ensemble** was tested, but did **not outperform** the best single model in this case

### Key Insight
Ensemble learning does **not always** produce better performance.  
Its effectiveness depends on:
- dataset quality
- model diversity
- model calibration

---

## 9. Recommended Presentation Flow
If presenting this project in class, use this order:

1. Introduce the problem
2. Explain dataset split (70/15/15)
3. Present the 3 individual models
4. Show comparison graphs
5. Explain ensemble evaluation
6. Conclude that ResNet50 is the best model
7. Demonstrate prediction using:
   - `predict.py`, or
   - `app_gradio.py`

---

## 10. Quick Commands Summary

### Train models
```bash
python train.py
```

### Evaluate ensemble
```bash
python evaluate_ensemble.py
```

### Run notebook
Open:
```text
summary_results_formal.ipynb
```

### Predict one image
```bash
python predict.py your_image.jpg
```

### Run web app
```bash
python app_gradio.py
```

---

## 11. Notes
- Image size used: `224 x 224`
- Classification type: **binary classification**
- Classes:
  - `cats`
  - `dogs`

---

# 🇹🇭 เวอร์ชันภาษาไทย

## 1. ภาพรวมโปรเจกต์
โปรเจกต์นี้เป็นการสร้างระบบ AI สำหรับจำแนกรูปภาพว่าเป็น **แมว** หรือ **หมา** โดยใช้ Deep Learning

โปรเจกต์นี้ไม่ได้มีแค่การเทรนโมเดลอย่างเดียว แต่ประกอบด้วย:

- เทรนโมเดลทั้งหมด 3 ตัว
- เปรียบเทียบผลลัพธ์ของแต่ละโมเดล
- ทดลองใช้ Ensemble
- นำโมเดลไปใช้งานจริงผ่านการทำนายภาพและเว็บแอป

ดังนั้นโปรเจกต์นี้จึงเหมาะทั้งสำหรับ:
- ส่งอาจารย์
- ใช้สาธิตการทำงานของ AI จริง

---

## 2. ไฟล์หลักในโปรเจกต์

### ไฟล์หลัก
- `train.py`  
  ใช้สำหรับเทรนโมเดล **SimpleCNN, MobileNetV2 และ ResNet50** แล้วบันทึกผลลัพธ์และโมเดลที่เทรนแล้ว

- `evaluate_ensemble.py`  
  ใช้โหลดโมเดลทั้ง 3 ตัวที่เทรนแล้ว เพื่อประเมินผลแบบ **Ensemble**

- `predict.py`  
  ใช้สำหรับทำนายรูปภาพจาก command line

- `app_gradio.py`  
  ใช้เปิดเว็บแอปสำหรับอัปโหลดรูปแล้วให้ AI ทำนายผล

- `summary_results_formal.ipynb`  
  ใช้สรุปผล เปรียบเทียบตาราง กราฟ และข้อสรุปอย่างเป็นทางการ

- `requirements.txt`  
  เก็บรายชื่อแพ็กเกจที่ต้องใช้ในโปรเจกต์

---

## 3. โครงสร้างโปรเจกต์

```text
DOG_CAT_AI/
│
├── train.py
├── evaluate_ensemble.py
├── predict.py
├── app_gradio.py
├── summary_results_formal.ipynb
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── val/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── results/
│   ├── summary_results.json
│   ├── ensemble_results.json
│   ├── SimpleCNN_accuracy.png
│   ├── SimpleCNN_loss.png
│   ├── MobileNetV2_accuracy.png
│   ├── MobileNetV2_loss.png
│   ├── ResNet50_accuracy.png
│   ├── ResNet50_loss.png
│   ├── *_classification_report.json
│   ├── *_confusion_matrix.csv
│   └── ensemble_confusion_matrix.csv
│
└── saved_models/
    ├── SimpleCNN.keras
    ├── MobileNetV2.keras
    └── ResNet50.keras
```

---

## 4. รูปแบบ Dataset
Dataset ต้องแบ่งออกเป็น 3 ส่วน:

- `train`
- `val`
- `test`

ในแต่ละส่วนต้องมี 2 โฟลเดอร์คือ:

- `cats`
- `dogs`

ตัวอย่าง:

```text
dataset/train/cats/
dataset/train/dogs/
dataset/val/cats/
dataset/val/dogs/
dataset/test/cats/
dataset/test/dogs/
```

สัดส่วนที่แนะนำ:
- Train = **70%**
- Validation = **15%**
- Test = **15%**

---

## 5. การติดตั้ง

### ขั้นตอนที่ 1: สร้าง virtual environment
```bash
python -m venv venv
```

### ขั้นตอนที่ 2: เปิดใช้งาน virtual environment
Windows:
```bash
venv\Scripts\activate
```

### ขั้นตอนที่ 3: ติดตั้ง package ที่จำเป็น
```bash
pip install -r requirements.txt
```

---

## 6. ลำดับการทำงานทั้งหมดแบบ Step by Step

## ขั้นตอนที่ 1 — เทรนโมเดลทั้ง 3 ตัว
รันคำสั่ง:

```bash
python train.py
```

### สิ่งที่ `train.py` ทำ
ไฟล์นี้จะ:

1. โหลดข้อมูลจาก:
   - `dataset/train`
   - `dataset/val`
   - `dataset/test`

2. เทรนโมเดล:
   - `SimpleCNN`
   - `MobileNetV2`
   - `ResNet50`

3. ประเมินผลแต่ละโมเดลบน **test set**

4. บันทึก:
   - โมเดลไว้ใน `saved_models/`
   - กราฟไว้ใน `results/`
   - สรุปผลไว้ใน `results/summary_results.json`

### ไฟล์ที่ควรได้หลังจากรัน `train.py`
ใน `saved_models/`:
- `SimpleCNN.keras`
- `MobileNetV2.keras`
- `ResNet50.keras`

ใน `results/`:
- `summary_results.json`
- `SimpleCNN_accuracy.png`
- `SimpleCNN_loss.png`
- `MobileNetV2_accuracy.png`
- `MobileNetV2_loss.png`
- `ResNet50_accuracy.png`
- `ResNet50_loss.png`
- `*_classification_report.json`
- `*_confusion_matrix.csv`

---

## ขั้นตอนที่ 2 — ประเมิน Ensemble
รันคำสั่ง:

```bash
python evaluate_ensemble.py
```

### สิ่งที่ `evaluate_ensemble.py` ทำ
ไฟล์นี้จะ:

1. โหลดโมเดลที่เทรนแล้วจาก `saved_models/`
2. โหลด **test dataset**
3. ให้โมเดลทั้ง 3 ตัวทำนายผล
4. รวมผลเป็น **ensemble**
5. บันทึกผลลัพธ์ลงใน `results/`

### ไฟล์ที่ควรได้หลังจากรัน `evaluate_ensemble.py`
- `results/ensemble_results.json`
- `results/ensemble_classification_report.json`
- `results/ensemble_confusion_matrix.csv`

---

## ขั้นตอนที่ 3 — วิเคราะห์ผลใน Notebook
เปิดไฟล์:

```text
summary_results_formal.ipynb
```

จากนั้นกด **Run All**

### Notebook นี้จะแสดงอะไรบ้าง
- ภาพรวมโปรเจกต์
- workflow
- ตารางเปรียบเทียบผลลัพธ์
- การจัดอันดับโมเดล
- กราฟ Accuracy
- กราฟ Precision
- กราฟ Recall
- กราฟ F1-score
- กราฟเวลาเทรน
- เปรียบเทียบ best single model กับ ensemble
- สรุปผลสุดท้าย

---

## ขั้นตอนที่ 4 — ทำนายรูปภาพ 1 รูปผ่าน command line
รันคำสั่ง:

```bash
python predict.py your_image.jpg
```

### ตัวอย่าง
```bash
python predict.py dataset/test/cats/cat.1.jpg
```

### ผลลัพธ์ที่คาดว่าจะเห็น
```text
Prediction: cats
Confidence: 0.9614
SimpleCNN: 0.1130
MobileNetV2: 0.0021
ResNet50: 0.0006
```

### ความหมาย
- `Prediction` = ผลทำนายสุดท้าย
- `Confidence` = ความมั่นใจของผลลัพธ์
- ค่าของแต่ละโมเดลแสดง probability จากโมเดลเดี่ยวแต่ละตัว

---

## ขั้นตอนที่ 5 — เปิดเว็บแอป demo AI
รันคำสั่ง:

```bash
python app_gradio.py
```

แล้วเปิด:

```text
http://127.0.0.1:7860
```

### เว็บแอปนี้ทำอะไร
- ให้อัปโหลดรูปภาพ
- AI ทำนายว่าเป็น **แมว** หรือ **หมา**
- แสดงความมั่นใจของผลลัพธ์

ส่วนนี้คือ **deployment/demo** ของโปรเจกต์

---

## 7. สรุปโมเดลที่ใช้

### โมเดลที่ใช้
1. **SimpleCNN**
   - โมเดลพื้นฐาน
   - สร้างเองจากศูนย์
   - ใช้เป็น baseline

2. **MobileNetV2**
   - โมเดล pretrained
   - เบาและเร็ว
   - ให้ผลลัพธ์ดี

3. **ResNet50**
   - โมเดล pretrained ที่ลึกกว่า
   - ให้ผลลัพธ์ดีที่สุดในโปรเจกต์นี้

4. **Ensemble**
   - ใช้การรวมผลทำนายจากหลายโมเดล
   - ทดลองเพื่อดูว่าช่วยเพิ่มประสิทธิภาพหรือไม่

---

## 8. การตีความผลลัพธ์
จากผลลัพธ์ของโปรเจกต์นี้พบว่า:

- **ResNet50** ให้ผลดีที่สุด
- **MobileNetV2** ก็ให้ผลดีมากเช่นกัน
- **SimpleCNN** ให้ผลต่ำสุด แต่ยังสำคัญในฐานะ baseline
- **Ensemble** ถูกทดลองแล้ว แต่ยังไม่ชนะโมเดลเดี่ยวที่ดีที่สุดในกรณีนี้

### Insight สำคัญ
Ensemble ไม่ได้ทำให้ผลดีขึ้นเสมอไป  
ผลลัพธ์ขึ้นอยู่กับ:
- คุณภาพของ dataset
- ความต่างของโมเดล
- การ calibration ของโมเดล

---

## 9. รวมคำสั่งสำคัญแบบรวดเร็ว

### เทรนโมเดล
```bash
python train.py
```

### ประเมิน ensemble
```bash
python evaluate_ensemble.py
```

### เปิด notebook
เปิดไฟล์:
```text
summary_results_formal.ipynb
```

### ทำนายรูปเดียว
```bash
python predict.py your_image.jpg
```

### เปิดเว็บแอป
```bash
python app_gradio.py
```

---

## 10. หมายเหตุ
- ขนาดรูปที่ใช้: `224 x 224`
- ประเภทงาน: **binary classification**
- คลาสที่ใช้:
  - `cats`
  - `dogs`

---