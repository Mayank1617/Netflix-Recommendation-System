# Personalized Content Discovery — Netflix Prize Recommendation System

Submission for **Open Projects 2026** (Cultural Council, IIT Roorkee) — *Problem Statement 1: Recommendation Systems for Personalized Content Discovery*.

**Author:** Mayank Kumar Agrawal (23323021)

## Overview
A personalized movie recommendation system built on the **Netflix Prize Dataset**
(100M+ ratings, 480K users, 17.7K movies). The system learns user preferences,
predicts unseen ratings, generates ranked Top-K recommendations, and explains them.

## Approach
A progression of four models, compared head-to-head:
1. **Bias baseline** — global mean + regularized user/item biases (RMSE floor)
2. **Item-based collaborative filtering** — adjusted-cosine item similarity
3. **Matrix factorization (biased SVD)** — latent factors trained with SGD (primary model)
4. **Neural collaborative filtering** — MLP over embeddings (innovation extension)

Plus an **explainable recommendation** layer ("recommended because you liked A and B").

## Evaluation
- **RMSE** — rating-prediction accuracy
- **MAP@10** — ranking quality (a movie is relevant if its true rating ≥ 3.5)
- Also: Precision@10, Recall@10, NDCG@10, Coverage
- **Temporal train/test split** — hold out each user's most recent 20% of ratings

## Repository structure
```
.
├── netflix_recsys.ipynb     # Main runnable notebook (Colab) — the end-to-end demo
├── src/                      # Importable modules mirroring the notebook
│   ├── data.py               #   Data Processing Pipeline   (parse, subset, split, index)
│   ├── models.py             #   Model Training Pipeline     (baseline, ItemCF, MF, NCF)
│   ├── evaluate.py           #   Evaluation Scripts          (RMSE, MAP@10, ranking metrics)
│   └── recommend.py          #   Recommendation Generation   (Top-K + explanations)
├── report/                   # Technical report (PDF, ≤10 pages)  — Deliverable 1
├── slides/                   # Presentation (PDF, ≤8 slides)       — Deliverable 3
├── results/                  # eda.png, model_comparison.csv
├── requirements.txt
└── README.md
```

### Deliverable-component mapping
| Required component | Where |
|--------------------|-------|
| Data Processing Pipeline | `src/data.py` · notebook §1–3 |
| Model Training Pipeline | `src/models.py` · notebook §5 |
| Evaluation Scripts | `src/evaluate.py` · notebook §6 |
| Recommendation Generation Module | `src/recommend.py` · notebook §7–8 |
| Documentation | this README + in-notebook markdown + report |

## Reproducing the results
**Easiest (recommended):** open `netflix_recsys.ipynb` in [Google Colab](https://colab.research.google.com)
(T4 GPU runtime), then **Run all**:
1. Get a Kaggle API token: kaggle.com → Settings → *Legacy API Credentials* → Create Legacy API Key (downloads `kaggle.json`).
2. Upload `kaggle.json` when the first cell prompts. The dataset downloads automatically.
3. The notebook parses the data, builds the subset, trains all four models, and reports RMSE + MAP@10.

**Locally / as scripts:** `pip install -r requirements.txt`, then import the `src/` modules
(see each module's docstring). A GPU is recommended for the MF/NCF models.

### Reproducibility notes
- Fixed random seed (`42`) for user sampling and both neural models.
- Deterministic data prep (temporal split, fixed Kaggle snapshot, parameters declared in-cell).
- The only non-determinism is sub-0.005 RMSE jitter on the GPU neural models (inherent to CUDA);
  model ranking and conclusions are stable across runs.

> **Subset note:** a dense subset (top 3,000 movies × 40,000 sampled active users ≈ 10.7M ratings,
> ~9% density) is used. This is permitted by the problem statement, trains in minutes, and yields
> stronger RMSE/MAP@10 by reducing noise from cold users/items. Criteria are documented in the notebook.
