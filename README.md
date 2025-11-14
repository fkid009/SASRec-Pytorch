# **SASRec-Pytorch**

PyTorch implementation of **SASRec (Self-Attentive Sequential Recommendation)**
based on the paper for studying sequential recommender system:

> **Self-Attentive Sequential Recommendation (SASRec)**
> *Wang-Cheng Kang, Julian McAuley (ICDM 2018)*
> [https://arxiv.org/abs/1808.09781](https://arxiv.org/abs/1808.09781)

---

## 📂 **Project Structure**

```
SASRec-Pytorch/
│
├── model/
│   └── sasrec.py        # SASRec model, evaluation, trainer
│
├── src/
│   ├── data.py          # data loader, sampler, batch generator
│   ├── path.py          # project path manager
│   ├── utils.py         # yaml loader, seed, misc utilities
│   └── main.py          # main training script
│
├── data/                # dataset directory (raw + processed)
│   └── ratings.csv      # Movielens input file
│
├── config.yaml          # all hyperparameters & experiment settings
└── README.md
```


---

All hyperparameters are centralized in `config.yaml`:

```yaml
data:
  dataset: "movielens"
  max_len: 50
  num_workers: 4

train:
  num_epochs: 100
  batch_size: 128
  num_batches_per_epoch: 1000
  device: "cuda"
  best_model_path: "./checkpoints/best_model.pth"

early_stopping:
  patience: 5
  min_delta: 0.0

model:
  hidden_dim: 64
  num_blocks: 2
  num_heads: 1
  dropout: 0.2
  first_norm: true

optimizer:
  name: "adam"
  lr: 0.001
  weight_decay: 0.0

loss:
  name: "bce_with_logits"
```


---

## **Run Training**

1. Place `ratings.csv` inside `data/`

```
data/
└── ratings.csv
```

2. Run main script:

```bash
uv run main.py
```

During training you’ll see:

```
[Epoch 10] Train Loss: 0.3421
  Val   - NDCG@10: 0.4821, Hit@10: 0.7013
  Test  - NDCG@10: 0.4751, Hit@10: 0.6942
  ** Best model updated and saved to './checkpoints/best_model.pth' **
```

---

## **Evaluation Metrics**

The repository implements the common metrics in sequential recommendation:

* **NDCG@10**
* **Hit Ratio@10**

Evaluation uses the standard:

* 1 positive item
* 100 negative samples

as in the original SASRec paper.

---

## **Typical Results (Movielens-1M, example)**

(Values vary depending on hyperparameters)

| Metric  | Score |
| ------- | ----- |
| NDCG@10 | ~0.48 |
| Hit@10  | ~0.70 |

---

## **Acknowledgements**

Inspired by the official SASRec implementations from:

* [https://github.com/pmixer/SASRec.pytorch/tree/main](https://github.com/pmixer/SASRec.pytorch/tree/main)

