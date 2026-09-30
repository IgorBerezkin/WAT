import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict

import torch

from wat.data import read_shakespeare, resolve_task, shakespeare_splits
from wat.run import atomic_json, config_group, environment


class NGram:
    def __init__(self, train, max_order, vocab):
        self.max_order = max_order
        seq = [int(t) for t in train]
        levels = [defaultdict(Counter) for _ in range(max_order)]
        for i in range(len(seq)):
            for k in range(min(max_order, i + 1)):
                levels[k][tuple(seq[i - k:i])][seq[i]] += 1
        self.levels = levels
        self.totals = [{ctx: sum(c.values()) for ctx, c in level.items()} for level in levels]
        self.best = [{ctx: c.most_common(1)[0][0] for ctx, c in level.items()} for level in levels]
        unigram = levels[0][()]
        self.base = [(unigram.get(c, 0) + 1) / (len(seq) + vocab) for c in range(vocab)]

    def prob(self, ctx, token, order):
        p = self.base[token]
        for k in range(1, min(len(ctx), order - 1) + 1):
            key = tuple(ctx[len(ctx) - k:])
            counts = self.levels[k].get(key)
            if counts is None:
                break
            types = len(counts)
            p = (counts.get(token, 0) + types * p) / (self.totals[k][key] + types)
        return p

    def argmax(self, ctx, order):
        for k in range(min(len(ctx), order - 1), -1, -1):
            guess = self.best[k].get(tuple(ctx[len(ctx) - k:]) if k else ())
            if guess is not None:
                return guess
        return 0


def evaluate(model, data, seq_len, order):
    seq = [int(t) for t in data]
    nll, correct, count = 0.0, 0, 0
    for start in range(0, len(seq) - seq_len - 1, seq_len):
        for j in range(start + 1, start + seq_len + 1):
            ctx = seq[max(start, j - order + 1):j]
            nll -= math.log2(model.prob(ctx, seq[j], order))
            correct += model.argmax(ctx, order) == seq[j]
            count += 1
    return {"bpc": nll / count, "acc": correct / count}


def main(argv=None):
    parser = argparse.ArgumentParser(description="n-gram anchors for character-level LM tasks.")
    parser.add_argument("--split", default="full", choices=["full", "paper"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--orders", default="1-8")
    parser.add_argument("--out", default="results/runs")
    args = parser.parse_args(argv)
    lo, _, hi = args.orders.partition("-")
    orders = range(int(lo), int(hi or lo) + 1)
    data, vocab = read_shakespeare()
    splits = shakespeare_splits(data, args.split)
    task = resolve_task({"name": "shakespeare", "split": args.split, "seq_len": args.seq_len})
    t0 = time.time()
    model = NGram(splits["train"], max(orders), vocab)
    build_time = time.time() - t0
    for order in orders:
        t1 = time.time()
        val = evaluate(model, splits["val"], args.seq_len, order)
        test = evaluate(model, splits["test"], args.seq_len, order)
        cfg = {"task": task, "model": {"name": "ngram", "order": order}, "seed": 0}
        name = f"shakespeare_ngram-n{order}_{config_group(cfg)}_s0"
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        atomic_json(os.path.join(args.out, name, "metrics.json"), {
            "status": "done", "name": name, "group": config_group(cfg), "config": cfg,
            "params": None, "contexts": sum(len(level) for level in model.levels[:order]),
            "embed_dim": None,
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
