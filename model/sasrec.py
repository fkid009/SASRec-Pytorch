import numpy as np
import random
from typing import Optional

import torch
import torch.nn as nn


# -----------------------------------------------------
# Point-wise FeedForward Network
# -----------------------------------------------------
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):  # inputs: (B, T, hidden_units)
        x = inputs.transpose(-1, -2)  # (B, hidden_units, T)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        outputs = x.transpose(-1, -2)  # (B, T, hidden_units)
        return outputs


# -----------------------------------------------------
# SASRec Model 
# -----------------------------------------------------
class SASRec(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        hidden_units,
        max_len,
        dropout_rate,
        num_blocks,
        num_heads,
        first_norm,
        device,
    ):
        super().__init__()

        # Multi-Head Attention 조건 체크
        assert hidden_units % num_heads == 0, "hidden_units는 num_heads로 나눠 떨어져야 합니다."

        self.user_num = user_num
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.first_norm = first_norm
        self.device = device

        # ---- Embeddings ----
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # ---- Transformer Blocks ----
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            # LN (Self-Attention)
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))

            # Multihead Self-Attention
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_units,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=False,  # (T, B, E)
                )
            )

            # LN (FFN)
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))

            # FeedForward
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

    # -------------------------------------------------
    # Embedding + Transformer → Hidden Sequence
    # -------------------------------------------------
    def log2feats(self, logs):
        if isinstance(logs, np.ndarray):
            logs_t = torch.LongTensor(logs).to(self.device)
        else:
            logs_t = logs.long().to(self.device)

        seqs = self.item_emb(logs_t)            # (B, T, H)
        seqs *= self.hidden_units ** 0.5        # Transformer scaling

        # Position embedding
        poss = np.tile(np.arange(1, logs_t.shape[1] + 1),
                       (logs_t.shape[0], 1))
        poss *= (logs_t.cpu().numpy() != 0)
        poss_t = torch.LongTensor(poss).to(self.device)

        seqs = seqs + self.pos_emb(poss_t)
        seqs = self.emb_dropout(seqs)

        # ---- Causal Mask ----
        T = seqs.shape[1]
        attn_mask = ~torch.tril(
            torch.ones((T, T), dtype=torch.bool, device=self.device)
        )

        # ---- Transformer Blocks ----
        for i in range(self.num_blocks):
            seqs = seqs.transpose(0, 1)  # (T, B, H)

            if self.first_norm:  # Pre-LN
                x = self.attention_layernorms[i](seqs)
                mha_out, _ = self.attention_layers[i](
                    x, x, x, attn_mask=attn_mask
                )
                seqs = seqs + mha_out
                seqs = seqs.transpose(0, 1)   # (B, T, H)

                seqs = seqs + self.forward_layers[i](
                    self.forward_layernorms[i](seqs)
                )

            else:                # Post-LN
                mha_out, _ = self.attention_layers[i](
                    seqs, seqs, seqs, attn_mask=attn_mask
                )
                seqs = self.attention_layernorms[i](seqs + mha_out)
                seqs = seqs.transpose(0, 1)

                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs)
                )

        return self.last_layernorm(seqs)  # (B, T, H)

    # -------------------------------------------------
    # 학습용 Forward
    # -------------------------------------------------
    def forward(self, seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(seqs)   # (B, T, H)

        pos_emb = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_emb = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        pos_logits = (log_feats * pos_emb).sum(dim=-1)
        neg_logits = (log_feats * neg_emb).sum(dim=-1)

        return pos_logits, neg_logits

    # -------------------------------------------------
    # 추론(Inference) for evaluation
    # -------------------------------------------------
    def predict(self, seqs, item_indices):
        if isinstance(seqs, np.ndarray):
            seqs_t = torch.LongTensor(seqs).to(self.device)
        else:
            seqs_t = seqs.long().to(self.device)

        item_idx_t = torch.LongTensor(item_indices).to(self.device)

        log_feats = self.log2feats(seqs_t)     # (B, T, H)
        final_feat = log_feats[:, -1, :]       # (B, H)

        item_embs = self.item_emb(item_idx_t)  # (N, H)

        # (B, H) @ (H, N) => (B, N)
        return final_feat @ item_embs.t()



# -----------------------------------------------------
# Evaluation (NDCG@10, Hit@10)
# -----------------------------------------------------
def evaluate(model, dataset, max_len, is_test=False):

    train, val, test, user_num, item_num = dataset.copy()

    NDCG, HT = 0.0, 0.0
    valid_user = 0

    users = (
        random.sample(range(1, user_num + 1), 10000)
        if user_num > 10000 else
        range(1, user_num + 1)
    )

    for u in users:
        if len(train[u]) < 1:
            continue
        if is_test and len(test[u]) < 1:
            continue
        if not is_test and len(val[u]) < 1:
            continue

        seq = np.zeros([max_len], dtype=np.int32)
        idx = max_len - 1

        if is_test:
            seq[idx] = val[u][0]
            idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)

        target = test[u][0] if is_test else val[u][0]

        item_idx = [target]
        for _ in range(100):
            t = np.random.randint(1, item_num + 1)
            while t in rated:
                t = np.random.randint(1, item_num + 1)
            item_idx.append(t)

        # Predict scores
        seq_arr = np.array([seq])
        arr_items = np.array(item_idx)

        scores = model.predict(seq_arr, arr_items)[0].detach().cpu().numpy()
        preds = -scores  # flip sign

        rank = preds.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    if valid_user == 0:
        return 0.0, 0.0

    return NDCG / valid_user, HT / valid_user



def trainer(
    model,
    sampler,
    optimizer,
    criterion,
    dataset,
    max_len: int,
    num_epochs: int,
    num_batch: int,
    eval_interval: int = 2,
    patience: int = 5,
    min_delta: float = 0.0,
    best_model_path: str = "best_model.pth",
):
    """
    Train loop with best-model saving and early stopping based on validation NDCG@10.

    Args:
        model: PyTorch model (must have `.device` attribute and support forward(seq, pos, neg)).
        sampler: WrapBatch-like object that has `next_batch()` method.
        optimizer: torch optimizer.
        criterion: loss function (e.g., BCEWithLogitsLoss).
        dataset: data tuple required by `evaluate(model, dataset, max_len, is_test=...)`.
        max_len: maximum sequence length.
        num_epochs: maximum number of epochs.
        num_batch: number of batches per epoch.
        eval_interval: evaluate every `eval_interval` epochs.
        patience: number of evaluations to wait for improvement before early stopping.
        min_delta: minimum improvement in validation NDCG to be considered as "better".
        best_model_path: file path to save the best model weights.
    """

    device = model.device

    # Best metrics tracking
    best_val_ndcg = float("-inf")
    best_val_hr = 0.0
    best_test_ndcg = 0.0
    best_test_hr = 0.0

    # Early stopping counter
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(num_batch):
            _, seq, pos, neg = sampler.next_batch()

            # Convert to numpy arrays (if not already)
            seq = np.array(seq)
            pos = np.array(pos)
            neg = np.array(neg)

            # Forward
            pos_logits, neg_logits = model(seq, pos, neg)

            # Binary labels for positive / negative logits
            pos_labels = torch.ones_like(pos_logits, device=device)
            neg_labels = torch.zeros_like(neg_logits, device=device)

            # Mask positions where pos == 0 (padding)
            mask = torch.tensor(pos != 0, dtype=torch.bool, device=device)

            optimizer.zero_grad()

            # Compute loss only on valid positions
            loss = criterion(pos_logits[mask], pos_labels[mask])
            loss += criterion(neg_logits[mask], neg_labels[mask])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / num_batch
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")

        # -------------------------------
        # Evaluation & early stopping
        # -------------------------------
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_ndcg, val_hr = evaluate(model, dataset, max_len, is_test=False)
                test_ndcg, test_hr = evaluate(model, dataset, max_len, is_test=True)

            print(f"  Val   - NDCG@10: {val_ndcg:.4f}, Hit@10: {val_hr:.4f}")
            print(f"  Test  - NDCG@10: {test_ndcg:.4f}, Hit@10: {test_hr:.4f}")

            # Check improvement on validation NDCG
            if val_ndcg > best_val_ndcg + min_delta:
                best_val_ndcg = val_ndcg
                best_val_hr = val_hr
                best_test_ndcg = test_ndcg
                best_test_hr = test_hr

                # Save best model weights
                torch.save(model.state_dict(), best_model_path)
                print(f"  ** Best model updated and saved to '{best_model_path}' **")

                epochs_without_improve = 0  # reset patience counter
            else:
                epochs_without_improve += 1
                print(f"  No improvement. Patience: {epochs_without_improve}/{patience}")

                # Early stopping condition
                if epochs_without_improve >= patience:
                    print("  >>> Early stopping triggered.")
                    break

    print("========================================")
    print(f"Best Validation : NDCG@10={best_val_ndcg:.4f}, Hit@10={best_val_hr:.4f}")
    print(f"Best Test       : NDCG@10={best_test_ndcg:.4f}, Hit@10={best_test_hr:.4f}")
    print(f"Best model weights saved at: {best_model_path}")
