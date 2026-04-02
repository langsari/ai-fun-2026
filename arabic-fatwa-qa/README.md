# Islamic Fatwa Hybrid Search — HOML End-to-End

> Arabic fatwa retrieval using DeepSeek + AraBERT — end-to-end in Google Colab with a laptop Flask UI

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)

---

## What This Project Does

This project builds a semantic search engine for Arabic Islamic fatwas using a hybrid of DeepSeek-R1-Distill-Qwen-1.5B and AraBERT embeddings. [file:1] You ask a question in Arabic, and the system ranks relevant fatwas from large Kaggle corpora by a weighted cosine similarity of both models. [file:1]

Inspired by the end-to-end pipeline style from “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” (HOML), the notebook goes from raw Kaggle CSVs → Arabic cleaning → hybrid embedding index → MRR@10 evaluation → a live API served from Colab and a local Flask UI on your laptop. [file:1]

---

## Project Structure

```text
fatwa-hybrid-search/
│
├── fatwa_search_Latest.ipynb   # Main Colab notebook (HOML-style pipeline)
├── laptop_server.py            # Flask UI that talks to Colab API
│
├── data/
│   ├── raw/                    # Downloaded Kaggle CSVs
│   └── processed/
│       ├── <dataset>_train.csv # Cleaned training split
│       └── <dataset>_test.csv  # Held-out evaluation split
│
├── models/
│   ├── deepseek/<dataset>/
│   │   └── embeddings_baseline.npy    # DeepSeek embeddings (frozen)
│   └── bert/<dataset>/
│       ├── embeddings_baseline.npy    # AraBERT baseline embeddings
│       └── embeddings_finetuned.npy   # AraBERT fine-tuned embeddings
│
├── reports/figures/
│   ├── eda_<dataset>.png       # Length histograms
│   └── mrr_<dataset>.png       # Baseline vs fine-tuned MRR@10
│
└── templates/
    └── index.html              # Simple Arabic search UI
```
[file:1]

Authors: Mohammed Yousef Salem, Tariq Saleh Alkindi. [file:1]

---

## Datasets

Three Kaggle datasets are supported via a small registry in the notebook. [file:1]

| Key         | Kaggle Dataset                                          | Size     | Columns                     |
|------------|----------------------------------------------------------|----------|-----------------------------|
| `islamweb` | `abdallahelsaadany/fatawa`                              | ~83K     | `title`, `ques`, `ans`      |
| `50k_mixed`| `hazemmosalah/50k-islamic-fatwa-q-and-a-dataset-arabic` | ~51K     | `question`, `answer`        |
| `binbaz`   | `a5medashraf/bin-baz-fatwas-dataset`                    | ~7K      | `Questions`, `Answers`      |
[file:1]

To switch datasets, change one line and re-run:

```python
ACTIVE_DATASET = "islamweb"  # or "50k_mixed" or "binbaz"
```
[file:1]

All downstream steps (cleaning, sampling, embeddings, MRR, UI) adapt automatically. [file:1]

---

## EDA — Data Distribution (IslamWeb ~83K)

On IslamWeb, basic EDA over question/answer lengths shows: [file:1]

| Metric                  | Questions | Answers |
|-------------------------|----------:|--------:|
| Median characters       |     247   |   880   |
| Median words            |      47   |   163   |
[file:1]

Key points:

- Questions are relatively short and focused; answers are much longer and richer. [file:1]  
- Both have long right tails, with some answers exceeding 5K words. [file:1]  
- Plots are saved to `reports/figures/eda_<dataset>.png`. [file:1]

---

## Pipeline Overview

The notebook follows a HOML-style end-to-end ML workflow: data, EDA, features, model, evaluation, serving. [file:1]

```text
Raw Kaggle CSV
      │
      ▼
ArabicCleaner        ← strip tashkeel, tatweel, non-Arabic chars
      │
      ▼
DFCleaner           ← build doc field from question / answer / title
      │
      ▼
Train / Test Split  ← sample_n, test_size, seed, min lengths
      │
      ▼
Compute embeddings (DeepSeek, AraBERT)
      │
      ▼
Hybrid search engine  (DeepSeek + AraBERT cosine)
      │
      ▼
MRR@10 evaluation  (anti-leakage: train index, test queries)
      │
      ▼
Colab Flask API + ngrok + laptop Flask UI
```

Core components:

- `ArabicCleaner`: removes diacritics, tatweel, and non-Arabic Unicode range using `pyarabic`. [file:1]  
- `DFCleaner`: builds cleaned columns and filters out very short questions/answers; saves processed train/test CSVs. [file:1]  
- `HybridSearchEngine`: wraps precomputed DeepSeek and AraBERT embeddings and returns top-k results with scores. [file:1]

---

## Hybrid Architecture

### Models

- DeepSeek encoder: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (frozen, used as a sentence encoder). [file:1]  
- AraBERT encoder: `aubmindlab/bert-base-arabertv02` via `sentence-transformers`. [file:1]

The encoder class for DeepSeek tokenizes with `AutoTokenizer` and averages token embeddings over the attention mask, in batches respecting a `ft_max_length` cap for Colab GPU. [file:1]

### Search Engine

Each fatwa question is embedded twice (DeepSeek and AraBERT) on a sampled corpus (`sample_n`, default 4000). [file:1]

At query time:

1. Encode query with DeepSeek and AraBERT.  
2. Compute cosine similarity against stored question embeddings for each model.  
3. Combine similarities with configurable weights: `hybrid_ara_weight` and `hybrid_deepseek_weight` (default 0.5 / 0.5). [file:1]  
4. Return top-k results with question, answer, and a `confidence` score. [file:1]

MRR@10 is computed on a held-out test subset by using a simple “answer prefix” mapping as ground truth and averaging reciprocal ranks over random samples. [file:1]

---

## Configuration

All important knobs live in a `CFG` dictionary in the notebook. [file:1]

| Parameter            | Default                                       | Meaning                       |
|----------------------|-----------------------------------------------|-------------------------------|
| `sample_n`           | 4000                                          | Rows used for embeddings      |
| `test_size`          | 0.10                                          | Test fraction                 |
| `seed`               | 42                                            | Random seed                   |
| `min_q`, `min_a`     | 5, 10                                         | Min lengths (chars)           |
| `deepseek_model`     | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`   | DeepSeek model ID             |
| `arabert_model`      | `aubmindlab/bert-base-arabertv02`             | AraBERT model ID              |
| `hybrid_ara_weight`  | 0.5                                           | Weight on AraBERT cosine      |
| `hybrid_deepseek_weight` | 0.5                                      | Weight on DeepSeek cosine     |
| `deepseek_batch`     | 1                                             | DeepSeek batch size (T4-safe) |
| `arabert_batch`      | 32                                            | AraBERT batch size            |
| `ft_epochs`          | 1                                             | Fine-tuning epochs            |
| `ft_batch`           | 16                                            | FT batch size                 |
| `ft_learning_rate`   | 2e-5                                          | FT learning rate              |
| `warmup`             | 0.10                                          | Warmup ratio                  |
| `ft_n`               | 900                                           | Pairs for AraBERT FT          |
| `ft_max_length`      | 512                                           | Max tokens for FT             |
| `top_k`              | 3                                             | Results per query             |
| `mrr_k`              | 10                                            | Cutoff for MRR@K              |
| `mrr_n`              | 300                                           | Eval sample size              |
[file:1]

This makes it easy to tune trade-offs between quality and speed while keeping the HOML-style configuration centralised. [file:1]

---

## Evaluation — MRR@10

The notebook reports baseline and fine-tuned AraBERT MRR@10 on a held-out test split: [file:1]

- Index is always built on `df_train`, and evaluation uses `df_test` only (strict anti-leakage). [file:1]  
- MRR@10 is averaged over a random sample (`mrr_n`) of test examples, with a simple exact-prefix based ground truth. [file:1]  
- A small bar plot comparing baseline vs fine-tuned MRR is saved to `reports/figures/mrr_<dataset>.png`. [file:1]

---

## Tech Stack

| Component       | Tool / Library                                   |
|----------------|---------------------------------------------------|
| Language       | Python 3.10 (Google Colab)                        |
| Data           | `pandas`, `kagglehub`                             |
| Arabic NLP     | `pyarabic`                                       |
| Embeddings     | `transformers`, `sentence-transformers`, `torch` |
| Metrics        | `scikit-learn` cosine similarity, custom MRR     |
| Plots          | `matplotlib`, `seaborn`                          |
| Serving (Colab)| `Flask`, `flask-cors`, `pyngrok`                 |
| Laptop UI      | `Flask`, `requests`                              |
[file:1]

---

## How to Run

### 1. Open Notebook in Colab

Upload `fatwa_search_Latest.ipynb` to Colab, select a GPU T4 runtime. [file:1]

### 2. Install Dependencies

The notebook includes a cell that installs everything:

```bash
pip install kagglehub sentence-transformers scikit-learn \
  pyarabic matplotlib seaborn scipy flask pyngrok flask-cors \
  transformers torch accelerate
```
[file:1]

### 3. Mount Google Drive (optional for persistence)

```python
from google.colab import drive
drive.mount('/content/drive')
```
[file:1]

### 4. Configure Dataset and CFG

- Set `ACTIVE_DATASET` in the dataset registry cell. [file:1]  
- Adjust `CFG` parameters (sample size, weights, batches) if needed. [file:1]

### 5. Run All

Use “Runtime → Run all”. The notebook will: [file:1]

1. Download the chosen Kaggle dataset via `kagglehub`.  
2. Clean Arabic text and build train/test splits.  
3. Compute DeepSeek and AraBERT embeddings on the training subset.  
4. Evaluate baseline and fine-tuned MRR@10.  
5. Plot EDA and MRR bar charts.  
6. Start a Flask API on Colab and expose it via ngrok.

### 6. Connect Laptop UI

The API cell prints something like: [file:1]

```text
Colab API live → https://xxxx.ngrok.io
1. Edit laptop_server.py → COLAB_URL = 'https://xxxx.ngrok.io'
```

On your laptop:

```bash
python laptop_server.py
```
[file:1]

Then visit:

- `http://localhost:8080` for the Arabic search UI. [file:1]  
- Type a question like `حكم صلاة الجمعة` and see ranked fatwas with confidence scores. [file:1]

---

## HOML Influence

The entire notebook is structured to mirror the HOML “end-to-end project” style: data registry, configurable `CFG`, careful data splits, clear evaluation metric, and a simple serving story that goes beyond the notebook into a working UI. [file:1] This makes it a practical case study of Arabic IR using DeepSeek and AraBERT framed in a HOML-inspired workflow. [file:1]

---

## License and Acknowledgements

- Code: MIT License (see repository).  
- Data: Respect original Kaggle dataset licenses. [file:1]  
- Models: DeepSeek-R1-Distill-Qwen-1.5B and AraBERT from their official providers. [file:1]  
- Structure and pipeline inspired by “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” (HOML). [file:1]
