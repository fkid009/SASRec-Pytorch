# main.py
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.path import ROOT_DIR, DATA_DIR
from src.data import data_load, data_split, WrapBatch
from src.utils import load_yaml, set_seed
from model.sasrec import SASRec, trainer


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Load Config
    # -------------------------------------------------------------------------
    CONFIG_FPATH = ROOT_DIR / "config.yaml"
    cfg = load_yaml(CONFIG_FPATH)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    es_cfg = cfg["early_stopping"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    loss_cfg = cfg["loss"]

    # -------------------------------------------------------------------------
    # Device & Seed
    # -------------------------------------------------------------------------
    req_dev = train_cfg.get("device", "cpu")
    if req_dev == "cuda" and not torch.cuda.is_available():
        print("[INFO] cuda is not available, switched to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(req_dev)
    train_cfg["device"] = str(device)

    set_seed(train_cfg.get("seed", 42))

    # Ensure data & checkpoint directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    best_model_path = train_cfg["best_model_path"]
    best_model_dir = os.path.dirname(best_model_path)
    if best_model_dir:
        os.makedirs(best_model_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
    dataset_name = data_cfg["dataset"]
    max_len = data_cfg["max_len"]

    # Preprocess raw data (user, item) and save to txt
    data_load(dataset_name)

    # Split into train / val / test sequences
    user_num, item_num, seq_train, seq_val, seq_test = data_split(dataset_name)
    dataset = [seq_train, seq_val, seq_test, user_num, item_num]

    # Multiprocessing sampler for (seq, pos, neg)
    sampler = WrapBatch(
        seq_train=seq_train,
        user_num=user_num,
        item_num=item_num,
        batch_size=train_cfg["batch_size"],
        max_len=max_len,
        n_workers=data_cfg["num_workers"],
    )

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------
    model = SASRec(
        user_num=user_num,
        item_num=item_num,
        hidden_units=model_cfg["hidden_dim"],
        max_len=max_len,
        dropout_rate=model_cfg["dropout"],
        num_blocks=model_cfg["num_blocks"],
        num_heads=model_cfg["num_heads"],
        first_norm=model_cfg.get("first_norm", True),
        device=device,
    ).to(device)

    # Ensure trainer can access model.device
    model.device = device

    # -------------------------------------------------------------------------
    # Optimizer & Loss
    # -------------------------------------------------------------------------
    opt_name = opt_cfg["name"].lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
    else:
        raise NotImplementedError(f"Optimizer '{opt_cfg['name']}' is not implemented.")

    loss_name = loss_cfg["name"].lower()
    if loss_name == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss '{loss_cfg['name']}' is not implemented.")

    # -------------------------------------------------------------------------
    # Training (with Early Stopping & Best Model Saving)
    # -------------------------------------------------------------------------
    trainer(
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        dataset=dataset,
        max_len=max_len,
        num_epochs=train_cfg["num_epochs"],
        num_batch=train_cfg["num_batches_per_epoch"],
        eval_interval=train_cfg["eval_interval"],
        patience=es_cfg["patience"],
        min_delta=es_cfg["min_delta"],
        best_model_path=train_cfg["best_model_path"],
    )

    # Clean up sampler workers
    sampler.close()
