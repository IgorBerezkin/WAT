import os
import urllib.request
import zipfile

import numpy as np
import torch


SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"

TASK_DEFAULTS = {
    "shakespeare": {"split": "full", "seq_len": 512},
    "enwik8": {"seq_len": 512},
    "copy": {"seq_len": 512, "n_mem": 16, "n_train": 6000, "n_val": 1000, "n_test": 1000,
             "data_seed": 42},
    "recall": {"seq_len": 256, "n_pairs": 12, "n_train": 6000, "n_val": 1000, "n_test": 1000,
               "data_seed": 42},
}


def resolve_task(cfg):
    if cfg["name"] not in TASK_DEFAULTS:
        raise ValueError(f"unknown task: {cfg['name']}")
    return {"name": cfg["name"], **TASK_DEFAULTS[cfg["name"]], **cfg}


def data_root():
    return os.environ.get("WAT_DATA_DIR", "data")


def read_shakespeare(root=None):
    root = root or data_root()
    path = os.path.join(root, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        urllib.request.urlretrieve(SHAKESPEARE_URL, tmp)
        os.replace(tmp, path)
    text = open(path, encoding="utf-8").read()
    index = {c: i for i, c in enumerate(sorted(set(text)))}
    return np.array([index[c] for c in text], dtype=np.int64), len(index)


def read_enwik8(root=None):
    root = root or data_root()
    path = os.path.join(root, "enwik8")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        archive = f"{path}.{os.getpid()}.zip"
        urllib.request.urlretrieve(ENWIK8_URL, archive)
        with zipfile.ZipFile(archive) as z:
            payload = z.read("enwik8")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
        os.remove(archive)
    return np.fromfile(path, dtype=np.uint8), 256


def enwik8_splits(data):
    return {"train": data[:90_000_000], "val": data[90_000_000:95_000_000],
            "test": data[95_000_000:100_000_000]}


def lm_splits(cfg):
    if cfg["name"] == "shakespeare":
        data, vocab = read_shakespeare()
        return shakespeare_splits(data, cfg["split"]), vocab
    data, vocab = read_enwik8()
    return enwik8_splits(data), vocab


def shakespeare_splits(data, split):
    if split == "full":
        n = len(data)
        a, b = int(n * 0.90), int(n * 0.95)
        return {"train": data[:a], "val": data[a:b], "test": data[b:]}
    if split == "paper":
        return {"train": data[:50_000], "test": data[50_512:55_512],
                "val": data[56_024:61_024]}
    raise ValueError(f"unknown shakespeare split: {split}")


class LMTask:
    def __init__(self, splits, vocab, seq_len):
        self.splits = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in splits.items()}
        self.vocab = vocab
        self.seq_len = seq_len

    def train_batch(self, batch_size, generator):
        data = self.splits["train"]
        starts = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,),
                               generator=generator)
        window = data[starts[:, None] + torch.arange(self.seq_len + 1)].long()
        return window[:, :-1], window[:, 1:]

    def eval_batches(self, split, batch_size, max_batches=None):
        data = self.splits[split]
        starts = torch.arange(0, len(data) - self.seq_len - 1, self.seq_len)
        offsets = torch.arange(self.seq_len + 1)
        for i, lo in enumerate(range(0, len(starts), batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            window = data[starts[lo:lo + batch_size, None] + offsets].long()
            yield window[:, :-1], window[:, 1:]


class SequenceTask:
    def __init__(self, splits, vocab):
        self.splits = splits
        self.vocab = vocab
        self.seq_len = splits["train"][0].size(1)

    def train_batch(self, batch_size, generator):
        x, y = self.splits["train"]
        idx = torch.randint(0, x.size(0), (batch_size,), generator=generator)
        return x[idx], y[idx]

    def eval_batches(self, split, batch_size, max_batches=None):
        x, y = self.splits[split]
        for i, lo in enumerate(range(0, x.size(0), batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            yield x[lo:lo + batch_size], y[lo:lo + batch_size]


def build_task(cfg):
    cfg = resolve_task(cfg)
    if cfg["name"] in ("shakespeare", "enwik8"):
        splits, vocab = lm_splits(cfg)
        return LMTask(splits, vocab, cfg["seq_len"])
    make = make_copy if cfg["name"] == "copy" else make_recall
    size_key = "n_mem" if cfg["name"] == "copy" else "n_pairs"
    splits, vocab = {}, None
    for offset, split in enumerate(("train", "val", "test")):
        x, y, vocab = make(cfg[f"n_{split}"], cfg["seq_len"], cfg[size_key], cfg["data_seed"] + offset)
        splits[split] = (x, y)
    return SequenceTask(splits, vocab)


def make_copy(n, seq_len, n_mem, seed):
    rng = np.random.RandomState(seed)
    V_CONTENT, NOISE, MARK = 16, 16, 17
    xs = np.full((n, seq_len), NOISE, dtype=np.int64)
    ys = np.full((n, seq_len), -100, dtype=np.int64)
    body = seq_len - n_mem
    for i in range(n):
        pos = np.sort(rng.choice(body, size=n_mem, replace=False))
        toks = rng.randint(0, V_CONTENT, size=n_mem)
        xs[i, pos] = toks
        xs[i, body:] = MARK
        ys[i, body:] = toks
    return torch.from_numpy(xs), torch.from_numpy(ys), 18


def make_recall(n, seq_len, n_pairs, seed):
    rng = np.random.RandomState(seed)
    N_KEYS, N_VALS = 16, 16
    KEY0, VAL0 = 0, N_KEYS
    NOISE, MARK = 32, 33
    xs = np.full((n, seq_len), NOISE, dtype=np.int64)
    ys = np.full((n, seq_len), -100, dtype=np.int64)
    body = seq_len - 2
    slots = np.arange(0, body - 1, 2)
    for i in range(n):
        keys = rng.permutation(N_KEYS)[:n_pairs]
        vals = rng.randint(0, N_VALS, size=n_pairs)
        pos = rng.choice(len(slots), size=n_pairs, replace=False)
        for k, v, p in zip(keys, vals, slots[pos]):
            xs[i, p] = KEY0 + k
            xs[i, p + 1] = VAL0 + v
        qi = rng.randint(0, n_pairs)
        xs[i, body] = MARK
        xs[i, body + 1] = KEY0 + keys[qi]
        ys[i, body + 1] = VAL0 + vals[qi]
    return torch.from_numpy(xs), torch.from_numpy(ys), 34
