import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict

from src.path import DATA_DIR


# ============================================================
# Data Loading
# ============================================================

def _get_raw_data_path(data_type: str) -> Path:
    """Return the path of the raw data file for the given dataset."""
    data_type = data_type.lower()

    if data_type == "movielens":
        return DATA_DIR / "ratings.csv"
    elif data_type == "amazon":
        raise NotImplementedError("Amazon dataset is not supported yet.")
    else:
        raise NotImplementedError(f"Unsupported dataset type: {data_type}")


def data_load(data_type: str) -> None:
    """
    Load raw data, extract (user, item), sort by (user, timestamp),
    and save as a text file containing lines of "user item".

    Output file:
        {DATA_DIR}/{data_type}_data.txt
    """
    fpath = _get_raw_data_path(data_type)
    data_type = data_type.lower()

    df = pd.read_csv(fpath)

    # Normalize column names (Movielens format)
    df = df.rename(columns={"userId": "user", "movieId": "item"})

    df = df.sort_values(by=["user", "timestamp"], ascending=[True, True])

    out_path = DATA_DIR / f"{data_type}_data.txt"

    df[["user", "item"]].to_csv(
        out_path,
        index=False,
        header=False,
        sep=" ",
    )


# ============================================================
# Train / Validation / Test Split
# ============================================================

def data_split(
    data_type: str,
) -> Tuple[int, int, Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Read the preprocessed "user item" file and split sequences by user.

    Split rule:
        - If user sequence length < 3:
            train = full sequence
            val = []
            test = []
        - Otherwise:
            train = all except last two items
            val = [second to last item]
            test = [last item]
    """
    data_type = data_type.lower()
    fpath = DATA_DIR / f"{data_type}_data.txt"

    user_num, item_num = 0, 0
    seq: DefaultDict[int, List[int]] = defaultdict(list)
    seq_train, seq_val, seq_test = {}, {}, {}

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            u_str, i_str = line.rstrip().split(" ")
            u, i = int(u_str), int(i_str)

            user_num = max(user_num, u)
            item_num = max(item_num, i)

            seq[u].append(i)

    for u, items in seq.items():
        if len(items) < 3:
            seq_train[u] = items
            seq_val[u] = []
            seq_test[u] = []
        else:
            seq_train[u] = items[:-2]
            seq_val[u] = [items[-2]]
            seq_test[u] = [items[-1]]

    return user_num, item_num, seq_train, seq_val, seq_test


# ============================================================
# Negative Sampling & Batch Generator
# ============================================================

def _random_neg(l: int, r: int, existing_items: set[int]) -> int:
    """
    Sample a negative item in the range [l, r) that is not in `existing_items`.
    (Simple rejection sampling.)
    """
    t = np.random.randint(l, r)
    while t in existing_items:
        t = np.random.randint(l, r)
    return t


def batch_function(
    seq_train: Dict[int, List[int]],
    user_num: int,
    item_num: int,
    batch_size: int,
    max_len: int,
    result_queue: Queue,
    seed: int,
) -> None:
    """
    Worker process that continuously generates batches and pushes them
    into the multiprocessing queue.
    """

    def sample(uid: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate (seq, pos, neg) arrays for the given user id.

        seq : input sequence of length max_len
        pos : positive target items
        neg : randomly sampled negative items
        """
        seq = np.zeros(max_len, dtype=np.int32)
        pos = np.zeros(max_len, dtype=np.int32)
        neg = np.zeros(max_len, dtype=np.int32)

        idx = max_len - 1
        nxt = seq_train[uid][-1]  # final item (positive target)

        item_pool = set(seq_train[uid])

        # Fill backward until max_len or sequence exhausted
        for item in reversed(seq_train[uid][:-1]):
            seq[idx] = item
            pos[idx] = nxt
            neg[idx] = _random_neg(1, item_num + 1, item_pool)

            nxt = item
            idx -= 1
            if idx < 0:
                break

        return uid, seq, pos, neg

    # Ensure reproducibility
    np.random.seed(seed)

    # User ids are assumed to be in [1, user_num]
    uids = np.arange(1, user_num + 1, dtype=np.int32)

    counter = 0
    while True:
        if counter % user_num == 0:
            np.random.shuffle(uids)

        batch = []
        for _ in range(batch_size):
            uid = uids[counter % user_num]
            batch.append(sample(uid))
            counter += 1

        # zip(*batch) → (uids, seqs, poss, negs)
        result_queue.put(tuple(zip(*batch)))


# ============================================================
# Batch Wrapper
# ============================================================

class WrapBatch:
    """
    Wrapper for multiprocessing-based batch generation.

    Example:
        wb = WrapBatch(seq_train, user_num, item_num, batch_size=128)
        uids, seqs, poss, negs = wb.next_batch()
    """

    def __init__(
        self,
        seq_train: Dict[int, List[int]],
        user_num: int,
        item_num: int,
        batch_size: int,
        max_len: int = 10,
        n_workers: int = 1,
    ) -> None:

        self.result_queue: Queue = Queue(maxsize=n_workers * 10)
        self.workers: List[Process] = []

        for _ in range(n_workers):
            p = Process(
                target=batch_function,
                args=(
                    seq_train,
                    user_num,
                    item_num,
                    batch_size,
                    max_len,
                    self.result_queue,
                    int(np.random.randint(2e9)),
                ),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def next_batch(self):
        """Retrieve one batch from the multiprocessing queue."""
        return self.result_queue.get()

    def close(self) -> None:
        """Terminate all worker processes."""
        for p in self.workers:
            p.terminate()
            p.join()
