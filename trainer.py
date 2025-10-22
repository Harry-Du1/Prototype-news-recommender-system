#!/usr/bin/env python3
"""
Trainer for a tiny, recency-aware news recommender on local MIND data.

• Uses a **local** MIND directory (e.g., `.../MINDlarge_train`) you already downloaded
• Parses `behaviors.tsv` & `news.tsv` (plus optional `entity_embedding.vec`, `relation_embedding.vec` — ignored by default)
• Builds a tiny tokenizer + vocab from news titles
• Trains the TinyNewsRec model (dual-encoder with ARIG user encoder)
• Evaluates AUC / MRR / nDCG@5/10 on the dev set

Usage:
    python train_tinynewsrec_mind.py \
        --data_path /path/to/MINDlarge_train \
        --workdir ./runs/mind_local \
        --epochs 1 --batch_size 256 --lr 2e-3 --max_title_len 20 --max_hist_len 50 \
        --dev_split 0.1  # if you only have train, this makes a time-aware 10% dev split

This script is self-contained (PyTorch + Pandas). If you have a GPU, set CUDA_VISIBLE_DEVICES.
"""

import os
import re
import io
import sys
import time
import json
import math
import shutil
import zipfile
import urllib.request
import argparse
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 1) Utils: download & setup
# ------------------------------
# (No remote download by default — we use --data_path)
SMALL_BASE = None
SMALL_TRAIN = None
SMALL_DEV = None

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1}
PAD_IDX = 0
UNK_IDX = 1

TIME_FMT = "%m/%d/%Y %I:%M:%S %p"  # e.g., 11/13/2019 8:36:57 AM


def download_if_needed(url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    local_zip = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(local_zip):
        print(f"Downloading {url} -> {local_zip}")
        urllib.request.urlretrieve(url, local_zip)
    else:
        print(f"Found existing {local_zip}, skipping download.")
    return local_zip


def unzip_to_dir(zip_path: str, out_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)


# ------------------------------
# 2) Tokenizer & Vocab
# ------------------------------
_token_pat = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _token_pat.findall(text.lower())


def build_vocab(titles: List[str], vocab_size: int = 50000) -> Dict[str, int]:
    freq = Counter()
    for t in titles:
        freq.update(tokenize(t))
    # most common minus special tokens
    itos = list(SPECIAL_TOKENS.keys()) + [w for w, _ in freq.most_common(vocab_size - len(SPECIAL_TOKENS))]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi


def numericalize(tokens: List[str], stoi: Dict[str, int], max_len: int) -> List[int]:
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens][:max_len]
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return ids

# ------------------------------
# 3) Model (same as provided earlier, made self-contained)
# ------------------------------

def masked_softmax(scores, mask, dim=-1):
    scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=dim)


class TinyTextCNN(nn.Module):
    def __init__(self, vocab_size, d_model=64, d_out=96, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding=k-1) for k in (2, 3, 4)
        ])
        self.proj = nn.Linear(3 * d_model, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, title_ids):  # (B, L)
        x = self.emb(title_ids)          # (B, L, d)
        x = x.transpose(1, 2)            # (B, d, L)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, kernel_size=h.size(-1)).squeeze(-1)  # (B, d)
            feats.append(h)
        h = torch.cat(feats, dim=-1)      # (B, 3d)
        h = self.proj(h)
        return self.ln(h)


class ItemEncoder(nn.Module):
    def __init__(self, vocab_size, d_title=96, pad_idx=0, d_out=128):
        super().__init__()
        self.title = TinyTextCNN(vocab_size, d_model=64, d_out=d_title, pad_idx=pad_idx)
        # 2 scalar features: popularity & age
        self.scalar_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(d_title + 16, d_out), nn.ReLU(), nn.LayerNorm(d_out)
        )

    def forward(self, title_ids, popularity=None, age_hours=None):
        B = title_ids.size(0)
        if popularity is None:
            popularity = torch.zeros(B, device=title_ids.device)
        if age_hours is None:
            age_hours = torch.zeros(B, device=title_ids.device)
        scalars = torch.stack([popularity, torch.log1p(age_hours)], dim=-1)
        return self.out(torch.cat([self.title(title_ids), self.scalar_mlp(scalars)], dim=-1))


class ARIGUserEncoder(nn.Module):
    def __init__(self, d_item=128, K_short=5):
        super().__init__()
        self.K_short = K_short
        self.decay_alpha = nn.Parameter(torch.tensor(0.05))
        self.att_q = nn.Linear(d_item, d_item, bias=False)
        self.att_k = nn.Linear(d_item, d_item, bias=False)
        self.att_v = nn.Linear(d_item, d_item, bias=False)
        self.gate = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.ln = nn.LayerNorm(d_item)

    def forward(self, hist_items, hist_mask, hist_age_hours=None, hist_popularity=None):
        B, T, d = hist_items.shape
        device = hist_items.device
        if hist_age_hours is None:
            hist_age_hours = torch.zeros(B, T, device=device)
        if hist_popularity is None:
            hist_popularity = torch.zeros(B, T, device=device)

        alpha = F.softplus(self.decay_alpha) + 1e-6
        decay = torch.exp(-alpha * hist_age_hours) * hist_mask.float()

        masked_items = hist_items * hist_mask.unsqueeze(-1)
        mean_hist = masked_items.sum(dim=1) / (hist_mask.sum(dim=1, keepdim=True) + 1e-6)

        q = self.att_q(mean_hist).unsqueeze(1)
        k = self.att_k(hist_items)
        v = self.att_v(hist_items)
        attn_scores = (q @ k.transpose(1, 2)) / math.sqrt(d)
        attn_scores = attn_scores + torch.log(decay.unsqueeze(1) + 1e-12)
        attn = masked_softmax(attn_scores, hist_mask.unsqueeze(1), dim=-1)
        long_term = (attn @ v).squeeze(1)

        # short-term: mean of last K true positions
        lastk_mask = torch.zeros_like(hist_mask)
        true_counts = hist_mask.sum(dim=1).clamp(min=1).long()
        for b in range(B):
            cnt = true_counts[b].item()
            if cnt > 0:
                start = max(0, cnt - self.K_short)
                lastk_mask[b, start:cnt] = True
        denom = lastk_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        short_term = (hist_items * lastk_mask.unsqueeze(-1)).sum(dim=1) / denom

        mean_pop = (hist_popularity * lastk_mask).sum(dim=1) / denom.squeeze(1)
        mean_rec = (hist_age_hours * lastk_mask).sum(dim=1) / denom.squeeze(1)
        g = self.gate(torch.stack([mean_pop, mean_rec], dim=-1))
        user = g * short_term + (1 - g) * long_term
        return self.ln(user)


class TinyNewsRec(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, d_item=128):
        super().__init__()
        self.item_encoder = ItemEncoder(vocab_size, d_title=96, pad_idx=pad_idx, d_out=d_item)
        self.user_encoder = ARIGUserEncoder(d_item=d_item, K_short=5)

    def encode_items(self, title_ids, popularity=None, age_hours=None):
        return self.item_encoder(title_ids, popularity, age_hours)

    def encode_users(self, hist_items, hist_mask, hist_age_hours=None, hist_popularity=None):
        return self.user_encoder(hist_items, hist_mask, hist_age_hours, hist_popularity)

    def score(self, user_vec, item_vec):
        return (user_vec * item_vec).sum(dim=-1)

    # In TinyNewsRec
    def info_nce_loss(self, user_vec, pos_item_vec, temperature=0.3):
        u = F.normalize(user_vec, dim=-1)
        v = F.normalize(pos_item_vec, dim=-1)
        logits = (u @ v.t()) / temperature
        labels = torch.arange(u.size(0), device=u.device)
        return F.cross_entropy(logits, labels)



# ------------------------------
# 4) Data loading for MIND-small
# ------------------------------
class MindData:
    def __init__(self, root: str, data_path: Optional[str] = None):
        self.root = root
        self.data_path = data_path  # path to directory that has behaviors.tsv, news.tsv
        self.train_dir = None
        self.dev_dir = None

    def prepare(self):
        os.makedirs(self.root, exist_ok=True)
        if self.data_path is not None:
            # User-provided local folder
            assert os.path.isdir(self.data_path), f"data_path not a directory: {self.data_path}"
            # Keep everything in one place; we'll split behaviors later if no dev present
            self.train_dir = self.data_path
            # If a sibling dev folder exists (e.g., MINDlarge_dev), use it
            maybe_dev = os.path.join(os.path.dirname(self.data_path), 'MINDlarge_dev')
            if os.path.isdir(maybe_dev) and os.path.exists(os.path.join(maybe_dev, 'behaviors.tsv')):
                self.dev_dir = maybe_dev
            print("Using local data_path:", self.data_path, " dev:", self.dev_dir)
        else:
            raise RuntimeError("Please provide --data_path pointing to a folder with behaviors.tsv & news.tsv")

    @staticmethod
    def read_news(news_path: str) -> pd.DataFrame:
        cols = ['id','category','subcategory','title','abstract','url','title_entities','abstract_entities']
        df = pd.read_table(news_path, header=None, names=cols, quoting=3)
        return df

    @staticmethod
    def read_behaviors(beh_path: str) -> pd.DataFrame:
        cols = ['impression_id','user_id','time','history','impressions']
        df = pd.read_table(beh_path, header=None, names=cols)
        return df

# ------------------------------
# 5) Building features (vocab, title ids, popularity, age)
# ------------------------------
class TitleFeaturizer:
    def __init__(self, stoi: Dict[str,int], max_len: int):
        self.stoi = stoi
        self.max_len = max_len

    def encode_title(self, title: str) -> List[int]:
        return numericalize(tokenize(title), self.stoi, self.max_len)


def build_feature_maps(train_news: pd.DataFrame, dev_news: pd.DataFrame,
                       train_beh: pd.DataFrame, dev_beh: pd.DataFrame,
                       max_title_len: int, vocab_size: int = 50000):
    # vocab from train+dev titles for simplicity
    stoi = build_vocab(pd.concat([train_news['title'], dev_news['title']]).fillna('').tolist(), vocab_size=vocab_size)
    tf = TitleFeaturizer(stoi, max_title_len)

    # map news_id -> title_ids
    def map_titles(df_news):
        return {nid: tf.encode_title(title) for nid, title in zip(df_news['id'], df_news['title'].fillna(''))}

    news2title = map_titles(train_news)
    news2title.update(map_titles(dev_news))

    # popularity: log1p(#appearances in any impression list)
    pop_counter = Counter()
    for df in (train_beh, dev_beh):
        for s in df['impressions']:
            if isinstance(s, str):
                for p in s.split():
                    if '-' in p:
                        nid, _ = p.split('-')
                        pop_counter[nid] += 1

    # first seen time (for age)
    news_first_ts: Dict[str, float] = {}
    def update_first_seen(df):
        for t, hist, imps in zip(df['time'], df['history'], df['impressions']):
            try:
                ts = datetime.strptime(t, TIME_FMT).timestamp()
            except Exception:
                continue
            # history items are definitely earlier than this impression
            if isinstance(hist, str):
                for nid in hist.split():
                    news_first_ts.setdefault(nid, ts)
            if isinstance(imps, str):
                for p in imps.split():
                    if '-' in p:
                        nid, _ = p.split('-')
                        news_first_ts.setdefault(nid, ts)
    update_first_seen(train_beh)
    update_first_seen(dev_beh)

    def pop_value(nid: str) -> float:
        return math.log1p(pop_counter.get(nid, 0))

    return stoi, tf, news2title, pop_value, news_first_ts


# ------------------------------
# 6) PyTorch Dataset
# ------------------------------
class MindTrainDataset(Dataset):
    def __init__(self, behaviors: pd.DataFrame, news2title: Dict[str, List[int]],
                 pop_value_fn, news_first_ts: Dict[str, float],
                 max_hist_len: int, max_title_len: int):
        self.max_hist_len = max_hist_len
        self.max_title_len = max_title_len
        self.news2title = news2title
        self.pop_value_fn = pop_value_fn
        self.news_first_ts = news_first_ts

        rows = []
        for _, row in behaviors.iterrows():
            imps = row['impressions']
            hist = row['history'] if isinstance(row['history'], str) else ''
            time_str = row['time']
            try:
                ts = datetime.strptime(time_str, TIME_FMT).timestamp()
            except Exception:
                ts = None
            if not isinstance(imps, str):
                continue

            # pick all clicked items (label=1)
            pos = [p.split('-')[0] for p in imps.split() if p.endswith('-1')]
            if len(pos) == 0:
                continue

            # filter history to items we can encode
            history_ids = [nid for nid in hist.split() if nid in self.news2title]
            # *** DROP zero-history samples ***
            if len(history_ids) == 0:
                continue

            # clip to max_hist_len (most recent last)
            history_ids = history_ids[-self.max_hist_len:]

            for nid in pos:
                if nid not in self.news2title:
                    continue
                rows.append((history_ids, nid, ts))
        self.rows = rows


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        hist_ids, pos_id, ts = self.rows[idx]
        # hist tensors
        hist_titles = [self.news2title[nid] for nid in hist_ids]
        hist_pop = [self.pop_value_fn(nid) for nid in hist_ids]
        hist_age = [max(0.0, (ts - self.news_first_ts.get(nid, ts or 0.0)) / 3600.0) if ts is not None else 0.0 for nid in hist_ids]

        # candidate
        pos_title = self.news2title[pos_id]
        pos_pop = self.pop_value_fn(pos_id)
        pos_age = max(0.0, (ts - self.news_first_ts.get(pos_id, ts or 0.0)) / 3600.0) if ts is not None else 0.0

        sample = {
            'hist_title_ids': torch.tensor(hist_titles, dtype=torch.long),   # (Th, L)
            'hist_pop': torch.tensor(hist_pop, dtype=torch.float),           # (Th,)
            'hist_age': torch.tensor(hist_age, dtype=torch.float),           # (Th,)
            'pos_title_ids': torch.tensor(pos_title, dtype=torch.long),      # (L,)
            'pos_pop': torch.tensor(pos_pop, dtype=torch.float),             # ()
            'pos_age': torch.tensor(pos_age, dtype=torch.float)              # ()
        }
        return sample


def collate_train(batch):
    # pad history to max T in batch
    max_T = max(x['hist_title_ids'].shape[0] for x in batch)
    L = batch[0]['pos_title_ids'].shape[0]
    B = len(batch)

    hist_title_ids = torch.full((B, max_T, L), PAD_IDX, dtype=torch.long)
    hist_mask      = torch.zeros((B, max_T), dtype=torch.bool)
    hist_pop       = torch.zeros((B, max_T), dtype=torch.float)
    hist_age       = torch.zeros((B, max_T), dtype=torch.float)

    pos_title_ids = torch.stack([x['pos_title_ids'] for x in batch])
    pos_pop       = torch.stack([x['pos_pop'] for x in batch])
    pos_age       = torch.stack([x['pos_age'] for x in batch])

    for i, x in enumerate(batch):
        h = x['hist_title_ids']
        h = torch.as_tensor(h, dtype=torch.long)

        # Ensure 2D shape (T, L) even when T=0 or when a single title is 1D
        if h.ndim == 1:
            if h.numel() == 0:
                h = torch.empty((0, L), dtype=torch.long)
            else:
                h = h[:L]
                if h.numel() < L:
                    h = torch.cat([h, torch.full((L - h.numel(),), PAD_IDX, dtype=torch.long)])
                h = h.view(1, L)
        else:
            h = h[:, :L]
            if h.size(1) < L:
                pad = torch.full((h.size(0), L - h.size(1)), PAD_IDX, dtype=torch.long)
                h = torch.cat([h, pad], dim=1)

        T = min(h.size(0), max_T)

        if T > 0:
            hist_title_ids[i, :T] = h[:T]
            hist_mask[i, :T]      = True

            # hist_pop/age expected to be length T; only assign if T>0
            hp = torch.as_tensor(x['hist_pop'], dtype=torch.float)
            ha = torch.as_tensor(x['hist_age'], dtype=torch.float)
            hist_pop[i, :T] = hp[:T] if hp.ndim == 1 else hp.view(-1)[:T]
            hist_age[i, :T] = ha[:T] if ha.ndim == 1 else ha.view(-1)[:T]

    return {
        'hist_title_ids': hist_title_ids,
        'hist_mask':      hist_mask,
        'hist_pop':       hist_pop,
        'hist_age':       hist_age,
        'pos_title_ids':  pos_title_ids,
        'pos_pop':        pos_pop,
        'pos_age':        pos_age,
    }



# ------------------------------
# 7) Evaluation utilities (dev impressions ranking)
# ------------------------------

def ndcg_at_k(labels: List[int], scores: List[float], k: int) -> float:
    # labels are 0/1 (can have multiple 1s)
    idx = np.argsort(scores)[::-1][:k]
    dcg = 0.0
    for rank, i in enumerate(idx, start=1):
        dcg += (2 ** labels[i] - 1) / math.log2(rank + 1)
    # ideal dcg
    sorted_labels = sorted(labels, reverse=True)[:k]
    idcg = 0.0
    for rank, lab in enumerate(sorted_labels, start=1):
        idcg += (2 ** lab - 1) / math.log2(rank + 1)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(labels: List[int], scores: List[float]) -> float:
    # rank of first clicked item
    order = np.argsort(scores)[::-1]
    for rank, i in enumerate(order, start=1):
        if labels[i] > 0:
            return 1.0 / rank
    return 0.0


def auc(labels: List[int], scores: List[float]) -> float:
    # simple AUC via rank
    pos_scores = [s for l, s in zip(labels, scores) if l == 1]
    neg_scores = [s for l, s in zip(labels, scores) if l == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan')
    wins = 0
    ties = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1
            elif ps == ns:
                ties += 1
    total = len(pos_scores) * len(neg_scores)
    return (wins + 0.5 * ties) / total if total > 0 else float('nan')


@torch.no_grad()
def evaluate_dev(model: TinyNewsRec, device: torch.device,
                 dev_beh: pd.DataFrame, news2title: Dict[str, List[int]],
                 pop_value_fn, news_first_ts: Dict[str, float],
                 max_title_len: int, max_hist_len: int,
                 batch_size_items: int = 1024) -> Dict[str, float]:
    model.eval()

    # Pre-encode all news that appear in dev
    all_dev_news = set()
    for s in dev_beh['impressions']:
        if isinstance(s, str):
            for p in s.split():
                if '-' in p:
                    nid, _ = p.split('-')
                    all_dev_news.add(nid)
    # build item tensors in batches
    news_vecs = {}
    news_ids = list(all_dev_news)
    N = len(news_ids)

    def encode_batch(nids: List[str]):
        titles = torch.tensor([news2title.get(n, [PAD_IDX]*max_title_len) for n in nids], dtype=torch.long, device=device)
        pops = torch.tensor([pop_value_fn(n) for n in nids], dtype=torch.float, device=device)
        # age will be filled per-impression; use 0 here and overwrite later by adding bias is messy.
        # For retrieval we can just use age=0 for items (user encoder uses history age correctly).
        ages = torch.zeros(len(nids), dtype=torch.float, device=device)
        return model.encode_items(titles, pops, ages).cpu()

    for i in range(0, N, batch_size_items):
        chunk = news_ids[i:i+batch_size_items]
        vecs = encode_batch(chunk)
        for nid, v in zip(chunk, vecs):
            news_vecs[nid] = v

    # iterate per-impression
    AUCs, MRRs, n5s, n10s = [], [], [], []
    for _, row in dev_beh.iterrows():
        imps = row['impressions']
        hist = row['history'] if isinstance(row['history'], str) else ''
        tstr = row['time']
        if not isinstance(imps, str):
            continue
        labels = []
        cand_ids = []
        for p in imps.split():
            if '-' in p:
                nid, lbl = p.split('-')
                if nid in news_vecs:
                    cand_ids.append(nid)
                    labels.append(1 if lbl == '1' else 0)
        if len(cand_ids) == 0:
            continue
        # user vector
        history_ids = [nid for nid in hist.split() if nid in news_vecs]
        history_ids = history_ids[-max_hist_len:]
        T = len(history_ids)
        if T == 0:
            # cold-start user; skip metric for this impression
            continue
        try:
            ts = datetime.strptime(tstr, TIME_FMT).timestamp()
        except Exception:
            ts = None
        # build tensors
        h_titles = torch.tensor([news2title.get(nid, [PAD_IDX]*max_title_len) for nid in history_ids], dtype=torch.long, device=device)
        h_pops   = torch.tensor([pop_value_fn(nid) for nid in history_ids], dtype=torch.float, device=device)
        h_ages   = torch.tensor([max(0.0, (ts - news_first_ts.get(nid, ts or 0.0)) / 3600.0) if ts is not None else 0.0 for nid in history_ids], dtype=torch.float, device=device)

        # encode hist items then user
        h_item_vecs = model.encode_items(h_titles)
        user_vec = model.encode_users(h_item_vecs.unsqueeze(0), torch.ones(1, T, dtype=torch.bool, device=device),
                                      h_ages.unsqueeze(0), h_pops.unsqueeze(0)).squeeze(0).cpu()

        # scores
        item_vecs = torch.stack([news_vecs[nid] for nid in cand_ids], dim=0)
        scores = (user_vec.unsqueeze(0) @ item_vecs.t()).squeeze(0).numpy().tolist()

        # metrics
        AUCs.append(auc(labels, scores))
        MRRs.append(mrr(labels, scores))
        n5s.append(ndcg_at_k(labels, scores, 5))
        n10s.append(ndcg_at_k(labels, scores, 10))

    def safe_mean(x):
        x = [v for v in x if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(x)) if x else float('nan')

    return {
        'AUC': safe_mean(AUCs),
        'MRR': safe_mean(MRRs),
        'nDCG@5': safe_mean(n5s),
        'nDCG@10': safe_mean(n10s)
    }


# ------------------------------
# 8) Train loop
# ------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Device:", device)

    data = MindData(args.workdir, data_path=args.data_path)
    data.prepare()

    # Load news & behaviors from local path(s)
    train_news = data.read_news(os.path.join(data.train_dir, 'news.tsv'))
    train_beh  = data.read_behaviors(os.path.join(data.train_dir, 'behaviors.tsv'))

    # If a dev folder exists, use it; otherwise time-aware split from train_beh
    if data.dev_dir and os.path.exists(os.path.join(data.dev_dir, 'behaviors.tsv')):
        dev_news = data.read_news(os.path.join(data.dev_dir, 'news.tsv'))
        dev_beh  = data.read_behaviors(os.path.join(data.dev_dir, 'behaviors.tsv'))
    else:
        # time-aware split: sort by timestamp and take last X% as dev
        print(f"No dev folder found — creating a time-aware split with ratio {args.dev_split}")
        beh = train_beh.copy()
        # Parse times safely; missing parse => push to train
        def _to_ts(x):
            try:
                return datetime.strptime(x, TIME_FMT).timestamp()
            except Exception:
                return 0.0
        beh['ts'] = beh['time'].apply(_to_ts)
        beh = beh.sort_values('ts')
        cut = int(len(beh) * (1.0 - args.dev_split))
        train_beh, dev_beh = beh.iloc[:cut].drop(columns=['ts']), beh.iloc[cut:].drop(columns=['ts'])
        dev_news = train_news  # reuse same news table

    stoi, tf, news2title, pop_value_fn, news_first_ts = build_feature_maps(
        train_news, dev_news, train_beh, dev_beh,
        max_title_len=args.max_title_len,
        vocab_size=args.vocab_size)

    ds_train = MindTrainDataset(train_beh, news2title, pop_value_fn, news_first_ts,
                                max_hist_len=args.max_hist_len, max_title_len=args.max_title_len)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                          collate_fn=collate_train, pin_memory=True)

    model = TinyNewsRec(vocab_size=len(stoi), pad_idx=PAD_IDX, d_item=args.d_model).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(dl_train)*args.epochs))

    best = {"AUC": 0.0}
    os.makedirs(args.workdir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        t0 = time.time()
        for step, batch in enumerate(dl_train, start=1):
            hist_title_ids = batch['hist_title_ids'].to(device)
            hist_mask = batch['hist_mask'].to(device)
            hist_pop = batch['hist_pop'].to(device)
            hist_age = batch['hist_age'].to(device)
            pos_title_ids = batch['pos_title_ids'].to(device)
            pos_pop = batch['pos_pop'].to(device)
            pos_age = batch['pos_age'].to(device)

            B, T, L = hist_title_ids.shape
            h_items = model.encode_items(hist_title_ids.view(B*T, L)).view(B, T, -1)
            user_vec = model.encode_users(h_items, hist_mask, hist_age, hist_pop)
            pos_item_vec = model.encode_items(pos_title_ids, pos_pop, pos_age)
            # --- DEBUG: logits stats + diag advantage ---
            with torch.no_grad():
                logits_dbg = (user_vec @ pos_item_vec.t()) / args.temperature
                diag = logits_dbg.diag().mean().item()
                off = (logits_dbg.sum() - logits_dbg.diag().sum()) / (logits_dbg.numel() - logits_dbg.size(0))
                if step % args.log_every == 0:
                    print(f"[dbg] logits mean diag={diag:.3f} off={off:.3f}  loss~{np.mean(losses[-args.log_every:]):.3f}")

            loss = model.info_nce_loss(user_vec, pos_item_vec, temperature=args.temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            losses.append(loss.item())
            if step % args.log_every == 0:
                print(f"Epoch {epoch} Step {step}/{len(dl_train)}  loss={np.mean(losses[-args.log_every:]):.4f}")

        dt = time.time() - t0
        print(f"Epoch {epoch} finished in {dt:.1f}s  avg_loss={np.mean(losses):.4f}")

        metrics = evaluate_dev(model, device, dev_beh, news2title, pop_value_fn, news_first_ts,
                               max_title_len=args.max_title_len, max_hist_len=args.max_hist_len)
        print("Dev metrics:", metrics)

        if metrics['AUC'] > best['AUC']:
            best = metrics
            torch.save({
                'model_state_dict': model.state_dict(),
                'stoi': stoi,
                'args': vars(args)
            }, os.path.join(args.workdir, 'tinynewsrec_mind_local_best.pt'))
            print("Saved best checkpoint.")

    print("Training done. Best dev:", best)


# ------------------------------
# 9) CLI
# ------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True, help='Path to local MIND folder containing behaviors.tsv & news.tsv (e.g., /.../MINDlarge_train)')
    p.add_argument('--workdir', type=str, default='./runs/mind_local', help='Where to save checkpoints and logs')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--max_title_len', type=int, default=20)
    p.add_argument('--max_hist_len', type=int, default=50)
    p.add_argument('--vocab_size', type=int, default=50000)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--dev_split', type=float, default=0.1, help='If no dev folder is found, take the last X fraction (by time) as dev')
    args = p.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train(args)
