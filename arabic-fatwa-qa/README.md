# 🕌 Islamic Fatwa Hybrid Search

> **Arabic fatwa retrieval using TF-IDF + AraBERT — fine-tuned end-to-end in Google Colab with a live Flask + ngrok UI**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)

---

## 📌 What This Project Does

This project builds a **semantic search engine** for Islamic fatwas (religious rulings) written in **Arabic**. You type a question in Arabic, and the system retrieves the most relevant fatwas from a large corpus — ranked by a combination of keyword matching (TF-IDF) and deep language understanding (AraBERT).

The full pipeline goes from raw Kaggle data → Arabic text cleaning → model training → evaluation → a live web UI accessible from your laptop.

---

## 🗂️ Project Structure

```
fatwa-hybrid-search/
│
├── fatwa_search.ipynb          # Main notebook (run in Google Colab)
├── laptop_server.py            # Auto-generated Flask UI for your laptop
│
├── data/
│   ├── raw/                    # Downloaded Kaggle CSVs
│   └── processed/
│       ├── islamweb_train.csv  # Cleaned training split
│       └── islamweb_test.csv   # Held-out evaluation split
│
├── models/
│   ├── tfidf/{dataset}/
│   │   ├── q_vec.pkl           # Fitted question TF-IDF vectorizer
│   │   ├── dv.pkl              # Fitted document TF-IDF vectorizer
│   │   ├── q_mat.npz           # Sparse question matrix
│   │   └── d_mat.npz           # Sparse document matrix
│   └── bert/{dataset}/
│       ├── embeddings.npy      # Pre-computed AraBERT embeddings
│       └── finetuned/          # Fine-tuned AraBERT model weights
│
├── reports/figures/
│   ├── eda_islamweb.png        # EDA histograms
│   └── mrr_islamweb.png        # MRR@10 bar chart
│
└── templates/
    └── index.html              # Auto-generated web UI template
```

---

## 📦 Datasets

Three Kaggle datasets are supported. Switch between them by changing one variable.

| Key | Kaggle Dataset | Size | Columns |
|---|---|---|---|
| `islamweb` *(default)* | `abdallahelsaadany/fatawa` | ~83K rows | `title`, `ques`, `ans` |
| `50k_mixed` | `hazemmosalah/50k-islamic-fatwa-q-and-a-dataset-arabic` | ~51K rows | `question`, `answer` |
| `binbaz` | `a5medashraf/bin-baz-fatwas-dataset` | ~7K rows | `Questions`, `Answers` |

**To switch datasets**, just change one line at the top of the notebook:
```python
ACTIVE_DATASET = "islamweb"   # change to "50k_mixed" or "binbaz"
```
Everything downstream (cleaning, training, evaluation, UI) updates automatically.

---

## 🔬 EDA — Data Distribution (IslamWeb ~83K)

Before training, the notebook runs an Exploratory Data Analysis on the raw dataset.

| Metric | Questions | Answers |
|---|---|---|
| Median Character Length | 247 chars | 880 chars |
| Median Word Count | 47 words | 163 words |

Key observations:
- Questions are short and focused (~47 words median)
- Answers are significantly longer (~163 words median), providing rich context for retrieval
- Both distributions have a long right tail — some entries are very long (capped at 500 chars / 2000 chars in plots for readability)
- Large spikes at the clip boundaries indicate a meaningful number of truncated-length entries

---

## ⚙️ Pipeline Overview

```
Raw CSV (Kaggle)
      │
      ▼
 ArabicCleaner          ← strips tashkeel, tatweel, non-Arabic chars
      │
      ▼
 DFCleaner              ← builds enriched "doc" field: title + question×3 + answer
      │
      ▼
 Train / Test Split     ← 90% train | 10% test | seed=42 | NO leakage
      │
      ├──────────────────────────────┐
      ▼                              ▼
 TF-IDF (2 vectorizers)       AraBERT Embeddings
 Q-vectorizer  (20K features)  aubmindlab/bert-base-arabertv02
 Doc-vectorizer (15K features)  batch_size=128
 n-grams: (1,4)
      │                              │
      └──────────┬───────────────────┘
                 ▼
          Hybrid Search Engine
          score = 0.5 × TF-IDF + 0.5 × BERT  (both min-max normalised)
                 │
                 ▼
         MRR@10 Evaluation
                 │
                 ▼
         Fine-Tune AraBERT (MNRL)
         → Re-evaluate MRR@10
                 │
                 ▼
         Flask API (Colab) + ngrok tunnel
         + Laptop Flask UI
```

---

## 🧠 Architecture Details

### Text Preprocessing
The `ArabicCleaner` transformer (scikit-learn compatible) does three things:
1. **Strips tashkeel** (Arabic diacritics) using `pyarabic`
2. **Strips tatweel** (elongation characters)
3. **Removes non-Arabic characters** (keeps only Unicode Arabic range `\u0600–\u06FF`)

The `DFCleaner` then builds a weighted `doc` field:
```
doc = title + (question × 3) + answer
```
This gives question tokens 3× more weight in TF-IDF scoring, which improves retrieval accuracy.

### Hybrid Search Score
For any query `q`:
```
tfidf_score = 0.25 × cosine(q, Q-matrix) + 0.75 × cosine(q, Doc-matrix)
bert_score  = cosine(encode(q), embeddings)
final_score = 0.50 × normalize(tfidf_score) + 0.50 × normalize(bert_score)
```
Both scores are **min-max normalised** before combining to ensure equal contribution.

### AraBERT Fine-Tuning
The model is fine-tuned using **Multiple Negatives Ranking Loss (MNRL)** from `sentence-transformers`:
- Training pairs: `(question_clean, answer_clean)` — treating each answer as the positive for its question
- 2,000 training pairs sampled from `df_train`
- 1 epoch, batch size 16, 10% warmup steps
- **No data from `df_test` is ever used during training** (strict anti-leakage)

---

## 📊 Results — MRR@10

The model is evaluated using **Mean Reciprocal Rank @ 10** on the held-out test set (500 random samples).

| Model | MRR@10 |
|---|---|
| Baseline (pre-trained AraBERT) | **0.0192** |
| Fine-Tuned AraBERT (MNRL) | **0.0197** |
| Δ Improvement | **+0.0005 ✅** |

> **What is MRR@10?** For each test question, we check if the correct answer appears in the top-10 results. If it ranks 1st, score = 1. If 2nd, score = 0.5. If 3rd, score = 0.33, etc. MRR is the average across all test questions. A higher score means the right answer appears higher in results.

> **Why is MRR low?** The corpus contains ~13,500 training fatwas. When a test question has an exact match in the index via only 80-character answer prefix lookup, MRR can be low because many questions have semantically similar but not identical fatwas. This is a known challenge in large-scale Arabic IR and reflects realistic retrieval difficulty.

---

## 🖥️ Configuration Reference

All hyperparameters live in the `CFG` dictionary for easy tuning:

| Parameter | Default | Description |
|---|---|---|
| `sample_n` | 15,000 | Max rows to use from dataset |
| `test_size` | 0.10 | Held-out test fraction |
| `seed` | 42 | Random seed for reproducibility |
| `min_q` / `min_a` | 5 / 10 | Min character length filter |
| `tfidf_q_max` | 20,000 | Max features for question vectorizer |
| `tfidf_d_max` | 15,000 | Max features for document vectorizer |
| `ngram` | (1,4) | N-gram range for TF-IDF |
| `min_df` | 2 | Minimum document frequency |
| `tfidf_q_w` / `tfidf_d_w` | 0.25 / 0.75 | TF-IDF sub-score weights |
| `hybrid_t` / `hybrid_b` | 0.50 / 0.50 | TF-IDF vs BERT blend |
| `bert_name` | `aubmindlab/bert-base-arabertv02` | HuggingFace model ID |
| `bert_batch` | 128 | Batch size for encoding |
| `ft_batch` | 16 | Batch size for fine-tuning |
| `ft_epochs` | 1 | Fine-tuning epochs |
| `ft_n` | 2,000 | Training pairs for fine-tuning |
| `warmup` | 0.10 | Warmup ratio |
| `top_k` | 3 | Results shown per search |
| `mrr_k` | 10 | K for MRR@K evaluation |
| `mrr_n` | 500 | Evaluation sample size |

---

## 🚀 How to Run

### Prerequisites
- Google account (for Colab + Drive)
- Kaggle account + API token (`kaggle.json`)
- ngrok account + authtoken (free at [ngrok.com](https://ngrok.com))

### Step 1 — Open in Google Colab
Upload `fatwa_search.ipynb` to [colab.research.google.com](https://colab.research.google.com) and set the runtime to **GPU (T4)**.

### Step 2 — Install Dependencies
The notebook installs all required packages automatically:
```bash
pip install kagglehub sentence-transformers scikit-learn pyarabic matplotlib seaborn scipy flask pyngrok flask-cors
```

### Step 3 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4 — Run All Cells
Click **Runtime → Run All**. The notebook will:
1. Download the dataset from Kaggle
2. Clean and split the data
3. Train TF-IDF vectorizers + compute AraBERT embeddings
4. Evaluate Baseline MRR@10
5. Fine-tune AraBERT with MNRL
6. Re-evaluate Fine-Tuned MRR@10
7. Start Flask API and expose it via ngrok

### Step 5 — Connect the Laptop UI
After Cell B runs, you will see:
```
Colab API live → https://xxxx.ngrok.io
```

1. Open the auto-generated `laptop_server.py`
2. Set `COLAB_URL = "https://xxxx.ngrok.io"`
3. Run locally:
```bash
python laptop_server.py
```
4. Open your browser at `http://localhost:8080`

---

## 🌐 API Reference

The Colab Flask API exposes two endpoints:

### `GET /health`
Returns device status.
```json
{ "status": "ok", "device": "cuda" }
```

### `POST /ask`
```json
{
  "question": "حكم صلاة الجمعة",
  "top_k": 3
}
```
Response:
```json
{
  "analysis": { "intent": "حكم صلاة الجمعة" },
  "results": [
    {
      "id": 1,
      "title": "...",
      "question": "...",
      "answer": "...",
      "confidence": "72.3%",
      "tfidf": "68.1%",
      "bert": "76.5%",
      "_idx": 4821
    }
  ]
}
```

---

## 🔧 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10 |
| Runtime | Google Colab (GPU T4) |
| Arabic NLP | `pyarabic`, `aubmindlab/bert-base-arabertv02` |
| Embeddings | `sentence-transformers` |
| Sparse Retrieval | `scikit-learn` TF-IDF |
| Fine-Tuning Loss | MultipleNegativesRankingLoss (MNRL) |
| API Server | Flask + flask-cors |
| Tunnel | pyngrok (ngrok) |
| Data Source | Kaggle (`kagglehub`) |
| Visualisation | matplotlib, seaborn |
| Storage | numpy `.npy`, scipy `.npz`, pickle `.pkl` |

---

## 📝 Key Design Decisions

**Why TF-IDF + BERT hybrid?**  
TF-IDF is fast and excellent at exact keyword matching. BERT captures semantic meaning even when words differ. Combining both gives better results than either alone, especially for religious text that mixes classical and modern Arabic.

**Why AraBERT specifically?**  
`aubmindlab/bert-base-arabertv02` is pre-trained on large Arabic corpora and handles Arabic morphology better than multilingual BERT. It understands Islamic terminology out of the box.

**Why MNRL for fine-tuning?**  
Multiple Negatives Ranking Loss treats all other answers in the same batch as negatives. This is very data-efficient — you only need `(question, answer)` pairs with no explicit negative labelling, making it ideal for fatwa datasets.

**Why strict train/test split?**  
The TF-IDF index and BERT embeddings are built **only on `df_train`**. MRR is measured **only on `df_test`**. This prevents data leakage and gives a realistic performance estimate.

---

## 🔮 Possible Improvements

- Use a larger fine-tuning sample (currently capped at 2,000 pairs)
- Try more epochs (currently 1) — limited by Colab free tier
- Add BM25 as a third retrieval signal
- Use a dedicated Arabic sentence transformer (e.g., `CAMeL-Lab` models)
- Add query expansion using Arabic synonyms
- Build a proper FAISS index for faster dense retrieval on large corpora
- Add multilingual support for non-Arabic users

---

## 📄 License

This project is released under the **MIT License**. Datasets are subject to their respective Kaggle licenses — please review before commercial use.

---

## 🙏 Acknowledgements

- [IslamWeb](https://www.islamweb.net) for the original fatwa content
- [AUBMindLab](https://github.com/aub-mind/arabert) for AraBERT
- [UKPLab](https://github.com/UKPLab/sentence-transformers) for sentence-transformers
- Kaggle dataset contributors: `abdallahelsaadany`, `hazemmosalah`, `a5medashraf`
- Inspired by *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (HOML) end-to-end project structure
