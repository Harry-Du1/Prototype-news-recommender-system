import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helpers ----------
def masked_softmax(scores, mask, dim=-1):
    scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=dim)

# ---------- Item Encoder ----------
class TinyTextCNN(nn.Module):
    """
    Title encoder: Embedding -> 1D conv (k=2,3,4) -> max-pool -> concat -> proj
    """
    def __init__(self, vocab_size, d_model=64, d_out=96, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding=k-1) for k in (2,3,4)
        ])
        self.proj = nn.Linear(3*d_model, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, title_ids):  # (B, L)
        x = self.emb(title_ids)                  # (B, L, d)
        x = x.transpose(1, 2)                    # (B, d, L)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))                  # (B, d, L')
            h = F.max_pool1d(h, kernel_size=h.size(-1)).squeeze(-1)  # (B, d)
            feats.append(h)
        h = torch.cat(feats, dim=-1)             # (B, 3d)
        h = self.proj(h)                         # (B, d_out)
        return self.ln(h)

class ItemEncoder(nn.Module):
    """
    Combines title TextCNN with optional entity/topic embeddings and 2 scalar features.
    Scalars: popularity (exposure propensity) and recency (hours since publish, log1p).
    """
    def __init__(self, vocab_size, ent_size=None, topic_size=None,
                 d_title=96, d_ent=32, d_topic=32, pad_idx=0, d_out=128):
        super().__init__()
        self.title = TinyTextCNN(vocab_size, d_model=64, d_out=d_title, pad_idx=pad_idx)
        self.has_ent = ent_size is not None
        self.has_topic = topic_size is not None

        if self.has_ent:
            self.ent_emb = nn.Embedding(ent_size, d_ent, padding_idx=0)
        if self.has_topic:
            self.topic_emb = nn.Embedding(topic_size, d_topic, padding_idx=0)

        # 2 small scalar features: popularity and age
        scalars_in = 2
        scalar_proj = 16

        in_dim = d_title + (d_ent if self.has_ent else 0) + (d_topic if self.has_topic else 0) + scalar_proj
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalars_in, scalar_proj),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.ReLU(),
            nn.LayerNorm(d_out)
        )

    def forward(self, title_ids, ent_ids=None, topic_ids=None, popularity=None, age_hours=None):
        """
        title_ids: (B, Lt)
        ent_ids:   (B, Le)  (multi-entities, will average)
        topic_ids: (B, ) or (B, Ltpc) as indices (if multi, average)
        popularity, age_hours: (B,)
        """
        B = title_ids.size(0)
        parts = [self.title(title_ids)]  # (B, d_title)

        if self.has_ent and ent_ids is not None:
            ent_vec = self.ent_emb(ent_ids)        # (B, Le, d_ent)
            ent_vec = ent_vec.mean(dim=1)          # (B, d_ent)
            parts.append(ent_vec)

        if self.has_topic and topic_ids is not None:
            if topic_ids.dim() == 2:
                topic_vec = self.topic_emb(topic_ids).mean(dim=1)  # (B, d_topic)
            else:
                topic_vec = self.topic_emb(topic_ids)               # (B, d_topic)
            parts.append(topic_vec)

        if popularity is None:
            popularity = torch.zeros(B, device=title_ids.device)
        if age_hours is None:
            age_hours = torch.zeros(B, device=title_ids.device)

        scalar_feats = torch.stack([popularity, torch.log1p(age_hours)], dim=-1)  # (B, 2)
        parts.append(self.scalar_mlp(scalar_feats))

        h = torch.cat(parts, dim=-1)
        return self.out(h)   # (B, d_out)

# ---------- User Encoder (ARIG) ----------
class ARIGUserEncoder(nn.Module):
    """
    Adaptive Recency-Interest Gate:
      - Long-term: attention over history (with learnable time decay)
      - Short-term: mean of last K clicks
      - Gate mixes them using small context (mean popularity & recency of recent clicks)
    """
    def __init__(self, d_item=128, K_short=5):
        super().__init__()
        self.K_short = K_short
        self.decay_alpha = nn.Parameter(torch.tensor(0.05))  # learns time decay (softplus-applied)
        self.att_q = nn.Linear(d_item, d_item, bias=False)   # attention query from global mean
        self.att_k = nn.Linear(d_item, d_item, bias=False)
        self.att_v = nn.Linear(d_item, d_item, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2, 1),  # mean_popularity, mean_recency
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(d_item)

    def forward(self, hist_items, hist_mask, hist_age_hours=None, hist_popularity=None):
        """
        hist_items: (B, T, d)
        hist_mask:  (B, T) boolean True=keep
        hist_age_hours, hist_popularity: (B, T)
        """
        B, T, d = hist_items.shape
        if hist_age_hours is None:
            hist_age_hours = torch.zeros(B, T, device=hist_items.device)
        if hist_popularity is None:
            hist_popularity = torch.zeros(B, T, device=hist_items.device)

        # Long-term with time decay
        alpha = F.softplus(self.decay_alpha) + 1e-6
        decay = torch.exp(-alpha * hist_age_hours)                      # (B, T)
        decay = decay * hist_mask.float()                               # mask out pads

        # Self-attention with a global query (mean of non-masked items)
        masked_items = hist_items * hist_mask.unsqueeze(-1)
        mean_hist = masked_items.sum(dim=1) / (hist_mask.sum(dim=1, keepdim=True) + 1e-6)  # (B, d)

        q = self.att_q(mean_hist).unsqueeze(1)                          # (B, 1, d)
        k = self.att_k(hist_items)                                      # (B, T, d)
        v = self.att_v(hist_items)                                      # (B, T, d)
        attn_scores = (q @ k.transpose(1, 2)) / math.sqrt(d)            # (B, 1, T)

        # Apply time-decay to attention logits additively
        attn_scores = attn_scores + torch.log(decay.unsqueeze(1) + 1e-12)  # (B, 1, T)
        attn = masked_softmax(attn_scores, hist_mask.unsqueeze(1), dim=-1) # (B, 1, T)
        long_term = attn @ v                                              # (B, 1, d)
        long_term = long_term.squeeze(1)

        # Short-term (last K valid items)
        idx = hist_mask.sum(dim=1).clamp(min=1).long()                   # (B,)
        # Gather last K indices per batch
        K = min(self.K_short, T)
        ar = torch.arange(T, device=hist_items.device).unsqueeze(0).expand(B, -1)  # (B, T)
        # Mask to keep last K true positions
        lastk_mask = torch.zeros_like(hist_mask)
        for b in range(B):
            cnt = idx[b].item()
            if cnt > 0:
                start = max(0, cnt - K)
                lastk_mask[b, start:cnt] = True
        short_items = hist_items * lastk_mask.unsqueeze(-1)
        denom = lastk_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        short_term = short_items.sum(dim=1) / denom                      # (B, d)

        # Gate with tiny context (mean popularity, mean recency of last K)
        mean_pop = (hist_popularity * lastk_mask).sum(dim=1) / denom.squeeze(1)
        mean_rec = (hist_age_hours * lastk_mask).sum(dim=1) / denom.squeeze(1)
        g = self.gate(torch.stack([mean_pop, mean_rec], dim=-1))         # (B, 1)
        user = g * short_term + (1 - g) * long_term
        return self.ln(user)                                             # (B, d)

# ---------- Full Model ----------
class TinyNewsRec(nn.Module):
    """
    Dual-encoder: user vector vs item vector, scored by dot product.
    Train with in-batch negatives via InfoNCE / sampled softmax.
    """
    def __init__(self, vocab_size, ent_size=None, topic_size=None, pad_idx=0, d_item=128):
        super().__init__()
        self.item_encoder = ItemEncoder(vocab_size, ent_size, topic_size,
                                        d_title=96, d_ent=32, d_topic=32, pad_idx=pad_idx, d_out=d_item)
        self.user_encoder = ARIGUserEncoder(d_item=d_item, K_short=5)
        self.topic_size = topic_size
        self.topic_size_is_scalar = topic_size is None

        # optional: project topic embedding to logits for diversity regularizer
        if topic_size is not None:
            self.topic_head = nn.Linear(d_item, topic_size)

    def encode_items(self, **item_kwargs):
        return self.item_encoder(**item_kwargs)  # (B, d)

    def encode_users(self, hist_items, hist_mask, hist_age_hours=None, hist_popularity=None):
        return self.user_encoder(hist_items, hist_mask, hist_age_hours, hist_popularity)  # (B, d)

    def score(self, user_vec, item_vec):
        return (user_vec * item_vec).sum(dim=-1)  # dot product

    def info_nce_loss(self, user_vec, pos_item_vec, temperature=0.07):
        """
        In-batch negatives: logits = U @ I^T
        """
        logits = (user_vec @ pos_item_vec.t()) / temperature             # (B, B)
        labels = torch.arange(user_vec.size(0), device=user_vec.device)  # (B,)
        return F.cross_entropy(logits, labels)

    def topic_diversity_regularizer(self, scores=None, topic_logits=None):
        """
        Encourage topic entropy to avoid same-topic collapse.
        Use either provided topic_logits (B, T) or derive soft weights from scores over batch (fallback).
        """
        if topic_logits is None:
            return torch.tensor(0., device=scores.device if scores is not None else 'cpu')
        probs = torch.softmax(topic_logits, dim=-1)       # (B, T)
        ent = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)  # (B,)
        return -ent.mean()  # negative entropy (to be added to loss with small lambda)

    def forward(self,
                hist_title_ids, hist_mask,
                hist_ent_ids=None, hist_topic_ids=None,
                hist_popularity=None, hist_age_hours=None,
                cand_title_ids=None, cand_ent_ids=None, cand_topic_ids=None,
                cand_popularity=None, cand_age_hours=None,
                return_loss=False, temperature=0.07, lambda_div=0.01):
        """
        Typical training batch: one positive candidate per user; negatives are other batch items.
        Shapes:
          hist_title_ids: (B, T, Lh), hist_mask: (B, T)
          cand_title_ids: (B, Lc)
        """
        B, T, Lh = hist_title_ids.shape
        # Encode history items (flatten -> encode -> reshape)
        h_items = self.item_encoder(
            title_ids=hist_title_ids.view(B*T, Lh),
            ent_ids=None if hist_ent_ids is None else hist_ent_ids.view(B*T, -1),
            topic_ids=None if hist_topic_ids is None else (
                hist_topic_ids.view(B*T) if hist_topic_ids.dim()==2 else hist_topic_ids.view(B*T, -1)
            ),
            popularity=None if hist_popularity is None else hist_popularity.view(B*T),
            age_hours=None if hist_age_hours is None else hist_age_hours.view(B*T),
        )  # (B*T, d)
        h_items = h_items.view(B, T, -1)

        user_vec = self.user_encoder(
            hist_items=h_items, hist_mask=hist_mask,
            hist_age_hours=hist_age_hours, hist_popularity=hist_popularity
        )  # (B, d)

        # Encode candidate items
        pos_item_vec = self.item_encoder(
            title_ids=cand_title_ids,
            ent_ids=cand_ent_ids, topic_ids=cand_topic_ids,
            popularity=cand_popularity, age_hours=cand_age_hours
        )  # (B, d)

        out = {
            "user_vec": user_vec,
            "pos_item_vec": pos_item_vec,
            "score": self.score(user_vec, pos_item_vec)
        }

        if return_loss:
            loss = self.info_nce_loss(user_vec, pos_item_vec, temperature=temperature)

            # Optional diversity regularizer via topic head
            if hasattr(self, "topic_head"):
                topic_logits = self.topic_head(pos_item_vec)  # (B, topic_size)
                loss = loss + lambda_div * self.topic_diversity_regularizer(topic_logits=topic_logits)

            out["loss"] = loss

        return out
