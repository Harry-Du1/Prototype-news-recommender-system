# Prototype-news-recommender-system
A news recommendation prototype trained on Microsoft news dataset. A personal project as I learn about recommendation systems. 

It demonstrates an end-to-end deep-learning pipeline for personalized ranking — from **data preprocessing and batching** to **training, inference, and evaluation**.

---

## ✨ Features
- Trains on the **MIND** news click dataset (small or large)
- **User click-history modeling** with padding + masks
- Robust to **cold-start users** (empty histories)
- **Negative sampling** for implicit-feedback training
- Clean **PyTorch** implementation (CPU / macOS MPS compatible)
- Time-aware **dev split** if no validation set is provided

---

## 📁 Requirements
- MIND train dataset download:https://msnews.github.io/
- python >= 3.9
- GPU preferred.

## Usage:
--batch_size — mini-batch size (e.g., 128–512)

--max_title_len — tokens per title (padding/cropping)

--max_hist_len — max clicked items per user

--dev_split — time-aware validation ratio if no dev set exists

```python
python trainer.py \
  --batch_size 256 \
  --max_title_len 20 \
  --max_hist_len 50 \
  --dev_split 0.1
```python

