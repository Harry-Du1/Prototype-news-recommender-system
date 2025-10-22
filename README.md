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
--data_path - path to the downloaded MIND dataset.

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

```
## Result:
Epoch 1 Step 200/11895  loss=0.89
Epoch 1 finished in 745.2s  avg_loss=0.92
Dev metrics: {'AUC': 0.67, 'MRR': 0.28, 'nDCG@5': 0.31, 'nDCG@10': 0.35}
Saved best checkpoint.

After this you should have an ensemble created in your working directory. 



