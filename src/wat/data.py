import os
import urllib.request

import numpy as np
import torch

from wat.lab import make_copy, make_recall

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


def data_root():
    return os.environ.get("WAT_DATA_DIR", "data")


def read_shakespeare(root=None):
    root = root or data_root()
    path = os.path.join(root, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = open(path, encoding="utf-8").read()
    index = {c: i for i, c in enumerate(sorted(set(text)))}
    return np.array([index[c] for c in text], dtype=np.int64), len(index)


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
        window = data[starts[:, None] + torch.arange(self.seq_len + 1)]
        return window[:, :-1], window[:, 1:]

    def eval_batches(self, split, batch_size, max_batches=None):
        data = self.splits[split]
        starts = torch.arange(0, len(data) - self.seq_len - 1, self.seq_len)
        offsets = torch.arange(self.seq_len + 1)
        for i, lo in enumerate(range(0, len(starts), batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            window = data[starts[lo:lo + batch_size, None] + offsets]
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
    name = cfg["name"]
    if name == "shakespeare":
        data, vocab = read_shakespeare()
        return LMTask(shakespeare_splits(data, cfg.get("split", "full")), vocab,
                      cfg.get("seq_len", 512))
    if name in ("copy", "recall"):
        seq_len = cfg.get("seq_len", 512 if name == "copy" else 256)
        seed = cfg.get("data_seed", 42)
        sizes = {"train": cfg.get("n_train", 6000), "val": cfg.get("n_val", 1000),
                 "test": cfg.get("n_test", 1000)}
        splits, vocab = {}, None
        for offset, split in enumerate(("train", "val", "test")):
            if name == "copy":
                x, y, vocab = make_copy(sizes[split], seq_len, cfg.get("n_mem", 16), seed + offset)
            else:
                x, y, vocab = make_recall(sizes[split], seq_len, cfg.get("n_pairs", 12), seed + offset)
            splits[split] = (x, y)
        return SequenceTask(splits, vocab)
    raise ValueError(f"unknown task: {name}")
