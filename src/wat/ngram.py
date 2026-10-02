import argparse
import os
import sys
import time

import numpy as np
import torch

from wat.data import lm_splits, resolve_task
from wat.run import atomic_json, config_group, environment


class NGram:
    def __init__(self, train, max_order, vocab):
        train = np.asarray(train, dtype=np.int64)
        if float(vocab) ** max_order >= 2 ** 63:
            raise ValueError("order too high for int64 context codes")
        self.vocab = vocab
        counts = np.bincount(train, minlength=vocab)
        self.base = (counts + 1) / (len(train) + vocab)
        self.base_best = int(counts.argmax())
        self.levels = [None]
        for k in range(1, max_order):
            ctx = np.zeros(len(train) - k, dtype=np.int64)
            for j in range(k):
                ctx = ctx * vocab + train[j:len(train) - k + j]
            pairs, pair_counts = np.unique(ctx * vocab + train[k:], return_counts=True)
            del ctx
            pair_ctx = pairs // vocab
            keys, first, types = np.unique(pair_ctx, return_index=True, return_counts=True)
            totals = np.add.reduceat(pair_counts, first)
            order = np.lexsort((pair_counts, pair_ctx))
            grouped = pair_ctx[order]
            last = np.r_[grouped[1:] != grouped[:-1], True]
            best = pairs[order][last] % vocab
            self.levels.append((keys, totals, types, pairs, pair_counts, best))

    def score(self, data, seq_len, order):
        data = np.asarray(data, dtype=np.int64)
        starts = np.arange(0, len(data) - seq_len - 1, seq_len)
        targets = (starts[:, None] + 1 + np.arange(seq_len)).ravel()
        avail = targets - np.repeat(starts, seq_len)
        y = data[targets]
        p = self.base[y]
        guess = np.full(len(y), self.base_best)
        active = np.arange(len(y))
        for k in range(1, order):
            active = active[avail[active] >= k]
            if not len(active):
                break
            code = np.zeros(len(active), dtype=np.int64)
            for j in range(k):
                code = code * self.vocab + data[targets[active] - k + j]
            keys, totals, types, pairs, pair_counts, best = self.levels[k]
            pos = np.minimum(np.searchsorted(keys, code), len(keys) - 1)
            hit = keys[pos] == code
            active, code, pos = active[hit], code[hit], pos[hit]
            pair = code * self.vocab + y[active]
            ppos = np.minimum(np.searchsorted(pairs, pair), len(pairs) - 1)
            count = np.where(pairs[ppos] == pair, pair_counts[ppos], 0)
            p[active] = (count + types[pos] * p[active]) / (totals[pos] + types[pos])
            guess[active] = best[pos]
        return {"bpc": float(-np.log2(p).mean()), "acc": float((guess == y).mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="n-gram anchors for character-level LM tasks.")
    parser.add_argument("--task", default="shakespeare", choices=["shakespeare", "enwik8"])
    parser.add_argument("--split", default="full", choices=["full", "paper"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--orders", default="1-6")
    parser.add_argument("--out", default="results/runs")
    args = parser.parse_args(argv)
    lo, _, hi = args.orders.partition("-")
    orders = range(int(lo), int(hi or lo) + 1)
    task = {"name": args.task, "seq_len": args.seq_len}
    if args.task == "shakespeare":
        task["split"] = args.split
    task = resolve_task(task)
    splits, vocab = lm_splits(task)
    t0 = time.time()
    model = NGram(splits["train"], max(orders), vocab)
    build_time = time.time() - t0
    for order in orders:
        t1 = time.time()
        val = model.score(splits["val"], args.seq_len, order)
        test = model.score(splits["test"], args.seq_len, order)
        cfg = {"task": task, "model": {"name": "ngram", "order": order}, "seed": 0}
        name = f"{args.task}_ngram-n{order}_{config_group(cfg)}_s0"
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        atomic_json(os.path.join(args.out, name, "metrics.json"), {
            "status": "done", "name": name, "group": config_group(cfg), "config": cfg,
            "params": None, "embed_dim": None,
            "result": {"val_bpc": val["bpc"], "val_acc": val["acc"],
                       "test_bpc": test["bpc"], "test_acc": test["acc"], "best_step": None},
            "train_time_s": round(build_time + time.time() - t1, 1),
            "env": environment(torch.device("cpu")),
        })
        print(f"n={order}: val_bpc={val['bpc']:.4f} val_acc={val['acc'] * 100:.2f}% "
              f"test_bpc={test['bpc']:.4f} test_acc={test['acc'] * 100:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
