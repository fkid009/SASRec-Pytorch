
import numpy as np, pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue

from src.path import DATA_DIR


def data_load(data_type: str):
    data_type = data_type.lower()

    if data_type== "movielens":
        FPATH = DATA_DIR / 'ratings.csv'
    elif data_type == "amazon":
        raise NotImplementedError
    else:
        raise NotImplementedError

    data = pd.read_csv(FPATH)
    data = data.rename(
            columns = {
                "userId": "user",
                "movieId": "item"

            }
        )

    data = data.sort_values(
        by = ["user", "timestamp"],
        ascending=[True, True]
    )

    data[["user", "item"]].to_csv(
        DATA_DIR / f"{data_type}_data.txt", 
        index = False,
        header = False,
        sep = " "
    )

def data_split(data_type: str):
    data_type = data_type.lower()

    FPATH = DATA_DIR / f"{data_type}_data.txt"

    user_num, item_num = 0, 0
    Seq = defaultdict(list)
    Seq_train, Seq_val, Seq_test = {}, {}, {}

    for line in open(FPATH, "r"):
        # print(line.rstrip.split(" "))
        u, i = line.rstrip().split(" ")
        u, i = int(u), int(i)
        user_num, item_num = max(user_num, u), max(item_num, i)
        Seq[u].append(i)
    
    for u in Seq:
        u_len = len(Seq[u])
        if u_len < 3:
            Seq_train[u] = Seq[u]
            Seq_val[u] = []
            Seq_test[u] = []
        else:
            Seq_train[u] = Seq[u][:-2]
            Seq_val[u] = [Seq[u][-2]]
            Seq_test[u] = [Seq[u][-1]]

    return user_num, item_num, Seq_train, Seq_val, Seq_test



def random_neg(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def batch_function(
    Seq_train, 
    user_num,
    item_num, 
    batch_size,
    max_len, 
    result_queue,
    SEED
):
    def sample(uid): # pos, neg are not implemented
        seq = np.zeros([max_len], dtype = np.int32) # 리스트를 씌우는 이유 -> shape을 명확하게 하기 위해서?
        pos = np.zeros([max_len], dtype = np.int32) # Seq + 마지막 정답값 
        neg = np.zeros([max_len], dtype = np.int32)

        idx = max_len - 1
        nxt = Seq_train[uid][-1] # 마지막 정답값

        item_pool = set(Seq_train[uid])
        for i in reversed(Seq_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neg(1, item_num + 1, item_pool)
            
            nxt = i
            idx -= 1
            if idx == -1: 
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, user_num + 1, dtype = np.int32) # 유저 전체 정보 -> 유저 max id 기준 (1 ~ max_id)으로 가정
    counter = 0 # for shuffle
    while True:
        if counter % user_num == 0: # user 리스트 셔플 트리거
            np.random.shuffle(uids)
        one_batch = []
        for _ in range(batch_size): # 배치만큼 추가 (batch_size, max_len)
            one_batch.append(sample(uids[counter % user_num]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WrapBatch:
    def __init__(self,
                Seq_train,
                user_num, 
                item_num,
                batch_size,
                max_len = 10,
                n_workers = 1):

        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target = batch_function, args = (
                    Seq_train,
                    user_num,
                    item_num,
                    batch_size,
                    max_len,
                    self.result_queue,
                    np.random.randint(2e9)
                ))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
