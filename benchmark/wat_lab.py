import argparse, json, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat import lab
from wat.lab import (CLSModel, CLSRootModel, LMDataset, LMModel, LSTMBackbone,
                     PaddedCLSDataset, TransformerBackbone, VARIANTS,
                     WATBackboneX, autocast_ctx, causality_probe,
                     make_brackets2, make_copy, make_depth, make_listops,
                     make_recall, match_embed_dim, n_params, train_model)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results", "benchmark", "lab")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

REF = {
    "copy": {"transformer": 51.66, "lstm": 6.34, "wat_v0 (прошлый прогон)": 6.46,
             "šance": 6.25},
    "lm": {"transformer @ep3": (31.27, 3.345), "lstm @ep3": (49.40, 2.465),
           "wat_v0 @ep3 (прошлый прогон)": (48.13, 2.517),
           "4-gram (test acc)": 42.60},
}



def load_shakespeare():
    return lab.load_shakespeare(DATA_DIR)


def build_backbone(name, vocab, cfg, max_len):
    if name in VARIANTS:
        kw = VARIANTS[name]
        def mk(ed):
            return WATBackboneX(vocab, ed, n_layers=cfg.n_layers,
                                chunk_size=32, max_len=max_len,
                                dropout=cfg.dropout, **kw)
    elif name == "transformer":
        def mk(ed):
            return TransformerBackbone(vocab, ed, n_layers=cfg.n_layers,
                                       max_len=max_len, dropout=cfg.dropout)
    elif name == "lstm":
        def mk(ed):
            return LSTMBackbone(vocab, ed, n_layers=cfg.n_layers,
                                dropout=cfg.dropout)
    else:
        raise ValueError(name)
    ed, _ = match_embed_dim(lambda e: LMModel(mk(e), vocab), cfg.target_params)
    return mk(ed), ed


def probe_or_skip(name, vocab, cfg, max_len):
    if name not in VARIANTS:
        return True
    kw = VARIANTS[name]
    bb = WATBackboneX(vocab, 32, n_layers=2, chunk_size=32, max_len=256,
                      dropout=0.0, **kw)
    ok, pos, mag = causality_probe(bb, vocab)
    if ok:
        print(f"  [{name}] causality probe: OK")
    else:
        print(f"  [{name}] !!! LEAK при p={pos} (|d|={mag:.2e}) — вариант "
              f"ПРОПУЩЕН !!!")
    return ok


def run_speed(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: SPEED (цена механизма)\n" + "=" * 78)
    device = cfg.device
    V = 65
    seqs = [256] if cfg.quick else [512, 2048]
    B = 2 if cfg.quick else 4
    rows = {}
    for name in cfg.variants:
        if not probe_or_skip(name, V, cfg, max(seqs)):
            rows[name] = {"LEAK": True}
            continue
        bb, ed = build_backbone(name, V, cfg, max(seqs))
        model = LMModel(bb, V).to(device)
        rows[name] = {"params": n_params(model), "toks_per_s": {}}
        for T in seqs:
            x = torch.randint(0, V, (B, T), device=device)
            y = torch.randint(0, V, (B, T), device=device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            try:
                for _ in range(2):
                    loss = F.cross_entropy(model(x).reshape(-1, V),
                                           y.reshape(-1))
                    loss.backward(); opt.step(); opt.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time(); iters = 3
                for _ in range(iters):
                    loss = F.cross_entropy(model(x).reshape(-1, V),
                                           y.reshape(-1))
                    loss.backward(); opt.step(); opt.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tps = B * T * iters / (time.time() - t0)
                rows[name]["toks_per_s"][T] = round(tps)
                print(f"  {name:<4} T={T:<5} {tps:>10,.0f} tok/s", flush=True)
            except torch.cuda.OutOfMemoryError:
                rows[name]["toks_per_s"][T] = None
                torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["speed"] = rows


def run_copy(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: COPY (ЗАМОРОЖЕННЫЙ протокол; главная метрика "
          "переноса)\n" + "=" * 78)
    device = cfg.device
    T = 128 if cfg.quick else 512
    ntr, nva = (300, 100) if cfg.quick else (6000, 1000)
    n_mem = 8 if cfg.quick else 16
    xtr, ytr, V = make_copy(ntr, T, n_mem, seed=42)
    xva, yva, _ = make_copy(nva, T, n_mem, seed=43)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.bs_lm,
                    shuffle=True, drop_last=True)
    vl = DataLoader(TensorDataset(xva, yva), batch_size=cfg.bs_lm)
    rows = {}
    for name in cfg.variants:
        if not probe_or_skip(name, V, cfg, T):
            rows[name] = {"LEAK": True}
            continue
        bb, ed = build_backbone(name, V, cfg, T)
        model = LMModel(bb, V)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_copy, cfg.lr,
                           "lm", name, patience=4)
        best["params"] = n_params(model)
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["copy"] = rows


def run_lm(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: LM-ЧЕК (3 эпохи, язык не сломан?)\n" + "=" * 78)
    device = cfg.device
    data, V = load_shakespeare()
    n = len(data)
    tr_end, va_end = int(n * 0.90), int(n * 0.95)
    train, val = data[:tr_end], data[tr_end:va_end]
    seq = 256 if cfg.quick else 512
    if cfg.quick:
        train, val = train[:60_000], val[:8_000]
    stride = seq if cfg.quick else 128
    tl = DataLoader(LMDataset(train, seq, stride), batch_size=cfg.bs_lm,
                    shuffle=True, drop_last=True)
    vl = DataLoader(LMDataset(val, seq, seq), batch_size=cfg.bs_lm)
    rows = {}
    for name in cfg.variants:
        if not probe_or_skip(name, V, cfg, seq):
            rows[name] = {"LEAK": True}
            continue
        bb, ed = build_backbone(name, V, cfg, seq)
        model = LMModel(bb, V)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_lm, cfg.lr,
                           "lm", name, patience=cfg.epochs_lm)
        best["params"] = n_params(model)
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["lm"] = rows


def _run_cls_task(cfg, results, key, title, xs, ys, V, PAD, max_len, extra=None,
                  muts=None):
    print("\n" + "=" * 78 + f"\nTASK: {title}\n" + "=" * 78)
    device = cfg.device
    ntr = int(len(xs) * 5 / 6)
    tl = DataLoader(PaddedCLSDataset(xs[:ntr], ys[:ntr], PAD, max_len),
                    batch_size=cfg.bs_cls, shuffle=True, drop_last=True)
    vl = DataLoader(PaddedCLSDataset(xs[ntr:], ys[ntr:], PAD, max_len),
                    batch_size=cfg.bs_cls)
    n_classes = len(set(ys))
    model_list = list(cfg.variants)
    if key in ("brackets2", "depth", "listops"):
        model_list = model_list + ["v0_root"]
    if cfg.baselines:
        model_list = model_list + ["transformer", "lstm"]
    rows = {}
    for name in model_list:
        base = "v0" if name == "v0_root" else name
        if not probe_or_skip(base, V, cfg, max_len):
            rows[name] = {"LEAK": True}
            continue
        bb, ed = build_backbone(base, V, cfg, max_len)
        model = (CLSRootModel if name == "v0_root" else CLSModel)(
            bb, n_classes, PAD)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_cls, cfg.lr,
                           "cls", name, patience=cfg.patience_cls)
        best["params"] = n_params(model)
        if muts is not None:
            model = model.to(device).eval()
            mut_val = muts[ntr:]
            by = {}
            with torch.no_grad():
                idx = 0
                for x, y in vl:
                    x = x.to(device)
                    with autocast_ctx(device):
                        pred = model(x).argmax(-1).cpu()
                    for b in range(x.size(0)):
                        m = mut_val[idx]
                        ok = int(pred[b].item() == ys[ntr + idx])
                        c, t = by.get(m, (0, 0))
                        by[m] = (c + ok, t + 1)
                        idx += 1
            best["by_difficulty"] = {
                ("bal" if m == 0 else f"{m}mut"): round(c / t * 100, 1)
                for m, (c, t) in sorted(by.items())}
            print(f"  [{name}] по сложности: {best['by_difficulty']}")
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if extra:
        rows["_info"] = extra
    results[key] = rows


def run_brackets2(cfg, results):
    lo, hi = (64, 128) if cfg.quick else (512, 1024)
    n = 480 if cfg.quick else 4800
    xs, ys, muts, V, PAD = make_brackets2(n, lo, hi, seed=cfg.seed)
    _run_cls_task(cfg, results, "brackets2",
                  f"BRACKETS v2 ({lo}-{hi}, смесь 1/2/4/8 мутаций)",
                  xs, ys, V, PAD, hi, muts=muts)


def run_depth(cfg, results):
    lo, hi = (64, 128) if cfg.quick else (256, 512)
    n = 480 if cfg.quick else 4800
    xs, ys, V, PAD, q = make_depth(n, lo, hi, seed=cfg.seed)
    _run_cls_task(cfg, results, "depth",
                  f"MAX NESTING DEPTH ({lo}-{hi}, 4 класса, квартили {q})",
                  xs, ys, V, PAD, hi, extra={"quartiles": q})


def run_listops(cfg, results):
    max_len = 128 if cfg.quick else 512
    n = 480 if cfg.quick else 4800
    xs, ys, V, PAD = make_listops(n, max_len, seed=cfg.seed)
    _run_cls_task(cfg, results, "listops",
                  f"LISTOPS (вложенные операции, len<={max_len}, 10 классов)",
                  xs, ys, V, PAD, max_len)


def run_recall(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: ASSOCIATIVE RECALL\n" + "=" * 78)
    device = cfg.device
    T = 64 if cfg.quick else 256
    ntr, nva = (300, 100) if cfg.quick else (6000, 1000)
    n_pairs = 4 if cfg.quick else 12
    xtr, ytr, V = make_recall(ntr, T, n_pairs, seed=cfg.seed)
    xva, yva, _ = make_recall(nva, T, n_pairs, seed=cfg.seed + 1)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.bs_lm,
                    shuffle=True, drop_last=True)
    vl = DataLoader(TensorDataset(xva, yva), batch_size=cfg.bs_lm)
    rows = {}
    model_list = list(cfg.variants) + (["transformer", "lstm"]
                                       if cfg.baselines else [])
    for name in model_list:
        if not probe_or_skip(name, V, cfg, T):
            rows[name] = {"LEAK": True}
            continue
        bb, ed = build_backbone(name, V, cfg, T)
        model = LMModel(bb, V)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_copy, cfg.lr,
                           "lm", name, patience=4)
        best["params"] = n_params(model)
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["recall"] = rows


def summarize(results):
    lines = ["# WAT LAB — итог", ""]
    if "speed" in results:
        seqs = sorted({t for r in results["speed"].values()
                       if "toks_per_s" in r for t in r["toks_per_s"]})
        lines += ["## Speed (tok/s)", "",
                  "| variant | " + " | ".join(f"T={t}" for t in seqs) + " |",
                  "|" + "---|" * (len(seqs) + 1)]
        for m, r in results["speed"].items():
            if "LEAK" in r:
                lines.append(f"| {m} | LEAK |")
                continue
            cells = [str(r["toks_per_s"].get(t, "-") or "OOM") for t in seqs]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    for key, title, is_lm in [("copy", "Selective copying (ГЛАВНАЯ)", True),
                              ("lm", "LM-чек (3 эпохи)", True),
                              ("brackets2", "Brackets v2", False),
                              ("depth", "Max depth", False),
                              ("listops", "ListOps", False),
                              ("recall", "Associative recall", True)]:
        if key not in results:
            continue
        lines += [f"## {title}", "",
                  "| variant | params | val acc | val bpc | best ep | time,s |",
                  "|---|---|---|---|---|---|"]
        for m, r in results[key].items():
            if m == "_info":
                continue
            if "LEAK" in r:
                lines.append(f"| {m} | — | LEAK | — | — | — |")
                continue
            lines.append(
                f"| {m} | {r.get('params', 0):,} | {r['val_acc']*100:.2f}% | "
                f"{r['val_bpc']:.3f} | {r['epoch']} | "
                f"{r['train_time_s']:.0f} |")
            if "by_difficulty" in r:
                lines.append(f"|  ↳ {m} по сложности: "
                             f"{r['by_difficulty']} |||||")
        if key in REF:
            lines.append("")
            lines.append(f"Референсы massive_benchmark: {REF[key]}")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all",
                   help="all | speed,copy,lm,brackets2,depth,listops,recall")
    p.add_argument("--variants", default="v0,v1,v2,v3,v4,v5")
    p.add_argument("--baselines", action="store_true",
                   help="добавить transformer/lstm на новых задачах")
    p.add_argument("--scale", default="base", choices=["small", "base", "big"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    cfg = p.parse_args()

    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True

    cfg.device = torch.device(cfg.device or
                              ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg.target_params, cfg.n_layers = {"small": (200_000, 2),
                                       "base": (1_000_000, 3),
                                       "big": (5_000_000, 4)}[cfg.scale]
    if cfg.quick:
        cfg.target_params, cfg.n_layers = 60_000, 2
    cfg.dropout, cfg.lr = 0.1, 3e-4
    cfg.bs_lm = 16 if cfg.device.type == "cpu" else 32
    cfg.bs_cls = 8 if cfg.device.type == "cpu" else 16
    cfg.epochs_copy = 1 if cfg.quick else 15
    cfg.epochs_lm = 1 if cfg.quick else 3
    cfg.epochs_cls = 1 if cfg.quick else 30
    cfg.patience_cls = 8
    cfg.variants = [v.strip() for v in cfg.variants.split(",") if v.strip()]
    tasks = (["speed", "copy", "lm", "brackets2", "depth", "listops", "recall"]
             if cfg.task == "all"
             else [t.strip() for t in cfg.task.split(",")])

    print("=" * 78)
    print("WAT LAB — варианты передачи контекста (архитектура: Igor Berezkin)")
    print(f"device={cfg.device}  scale={cfg.scale}  "
          f"target={cfg.target_params:,}  L={cfg.n_layers}")
    print(f"variants={cfg.variants}  tasks={tasks}  baselines={cfg.baselines}")
    print("=" * 78)

    results = {"config": {k: (str(v) if isinstance(v, torch.device) else v)
                          for k, v in vars(cfg).items()}}
    t0 = time.time()
    runners = {"speed": run_speed, "copy": run_copy, "lm": run_lm,
               "brackets2": run_brackets2, "depth": run_depth,
               "listops": run_listops, "recall": run_recall}
    for t in tasks:
        runners[t](cfg, results)
    results["total_time_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT_DIR, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    summary = summarize(results)
    with open(os.path.join(OUT_DIR, "summary.md"), "w",
              encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nГотово за {results['total_time_s']/60:.1f} мин. "
          f"results/benchmark/lab/results.json, summary.md")


if __name__ == "__main__":
    main()
