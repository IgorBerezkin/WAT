import json
import math
import statistics
from collections import defaultdict

from wat.report import collect
from wat.run import config_group, merge, resolve, set_path


def best_by_val(rows):
    groups = defaultdict(list)
    for metrics in rows:
        cfg = metrics["config"]
        train = {k: v for k, v in cfg["train"].items() if k != "lr"}
        key = json.dumps([cfg["task"], cfg["model"], train], sort_keys=True)
        groups[key].append(metrics)
    return [min(items, key=lambda m: m["result"]["val_bpc"]) for _, items in sorted(groups.items())]


def extra_seeds(runs_dir, seeds=(1, 2)):
    runs = []
    for m in best_by_val(collect(runs_dir)):
        cfg = m["config"]
        for seed in seeds:
            runs.append({"task": cfg["task"], "model": cfg["model"], "train": cfg["train"], "seed": seed})
    return {"runs": runs}


def promote(runs_dir, baseline, ref=3_000_000, target=10_000_000, layers=6, top=2):
    rows = [m for m in collect(runs_dir) if m["config"]["model"].get("target_params") == ref]

    def is_baseline(m):
        return m["config"]["train"]["lr"] == baseline["lr"] and \
            all(m["config"]["model"].get(k) == v for k, v in baseline["model"].items())

    bar = min(m["result"]["val_bpc"] for m in rows if is_baseline(m))
    better = sorted((m for m in rows if not is_baseline(m) and m["result"]["val_bpc"] < bar
                     and m["config"]["train"]["lr"] == baseline["lr"]),
                    key=lambda m: m["result"]["val_bpc"])[:top]
    runs = []
    for m in better:
        model = {k: v for k, v in m["config"]["model"].items() if k != "embed_dim"}
        model.update(target_params=target, n_layers=layers)
        runs.append({"task": m["config"]["task"], "model": model, "seed": m["config"]["seed"],
                     "train": m["config"]["train"]})
    return {"runs": runs}


def phase1_scaling(runs_dir, sizes=((10_000_000, 6), (3_000_000, 4)), lr_divisor=3,
                   seeds=(0, 1, 2), extra_seeds=(1, 2), seq_len=512, enwik8=None):
    schedule = {"steps": 6000, "warmup": 300, "eval_every": 1000, "eval_batches": 40,
                **(enwik8 or {})}
    task = {"name": "enwik8", "seq_len": seq_len}
    winners = [m for m in best_by_val(collect(runs_dir)) if m["config"]["task"]["name"] == "shakespeare"]
    runs = []
    for target, layers in sizes:
        for m in winners:
            model = {k: v for k, v in m["config"]["model"].items() if k != "embed_dim"}
            model.update(target_params=target, n_layers=layers)
            lr = m["config"]["train"]["lr"]
            for candidate in (lr, lr / lr_divisor):
                runs.append({"task": task, "model": model, "seed": seeds[0],
                             "train": dict(m["config"]["train"], lr=candidate, **schedule)})
    for m in winners:
        for seed in seeds:
            runs.append({"task": task, "model": m["config"]["model"], "seed": seed,
                         "train": dict(m["config"]["train"], **schedule)})
    for m in winners:
        for seed in extra_seeds:
            runs.append({"task": m["config"]["task"], "model": m["config"]["model"], "seed": seed,
                         "train": m["config"]["train"]})
    return {"runs": runs}


NIGHT_BASE = {"task": {"name": "enwik8", "seq_len": 512},
              "model": {"name": "wat", "ctx_mode": "tree", "intra": False, "chunk_size": 32,
                        "dropout": 0.1},
              "train": {"steps": 6000, "batch_size": 32, "lr": 0.001, "warmup": 300,
                        "eval_every": 1000, "eval_batches": 40, "eval_batch_size": 32}}
NIGHT_SIZES = {"1m": (1_000_000, 3), "3m": (3_000_000, 4), "10m": (10_000_000, 6)}
NIGHT_SINGLES = {
    "base": {},
    "gall": {"model.ctx_mode": "tree_gread", "model.read": "all"},
    "gread": {"model.ctx_mode": "tree_gread"},
    "deep": {"depth": 2},
    "sel": {"model.ctx_mode": "tree_sel"},
    "all": {"model.read": "all"},
    "dw15": {"model.conv": "depthwise", "model.conv_k": 15},
    "dw7": {"model.conv": "depthwise", "model.conv_k": 7},
    "dw4": {"model.conv": "depthwise", "model.conv_k": 4},
    "write": {"model.leaf": "gated"},
    "ctxnorm": {"model.ctx_norm": True},
    "merge2": {"model.merge2": True},
    "add": {"model.inject": "add"},
    "nopos": {"model.pos": "none"},
    "lr3": {"train.lr": 0.003},
    "drop0": {"model.dropout": 0.0},
}


def night_config(name, delta, size, seed=0, extra=None):
    target, layers = NIGHT_SIZES[size]
    cfg = merge(NIGHT_BASE, {"name": name, "seed": seed, "model": {
        "target_params": target, "n_layers": layers * delta.get("depth", 1)}})
    for key, value in {**delta, **(extra or {})}.items():
        if key != "depth":
            set_path(cfg, key, value)
    return cfg


def night_singles(runs_dir, prefix="n1", sizes=("3m", "1m")):
    return {"runs": [night_config(f"{prefix}-{tag}-{size}", delta, size)
                     for size in sizes for tag, delta in NIGHT_SINGLES.items()]}


def night_index(runs_dir):
    return {(m["config"].get("name"), m["config"]["seed"]): m for m in collect(runs_dir)
            if math.isfinite(m["result"]["val_bpc"])}


def night_val(index, name, seed=0):
    m = index.get((name, seed))
    return m["result"]["val_bpc"] if m else None


def night_gains(index, prefix):
    gains = {}
    for size in ("3m", "1m"):
        vals = {tag: night_val(index, f"{prefix}-{tag}-{size}") for tag in NIGHT_SINGLES}
        done = [v for v in vals.values() if v is not None]
        ref = vals["base"] if vals["base"] is not None else (statistics.median(done) if done else None)
        for tag, v in vals.items():
            if tag != "base":
                gains.setdefault(tag, {})[size] = None if v is None or ref is None else v - ref
    return gains


def night_positives(gains, threshold):
    ok = [t for t, g in gains.items() if g["3m"] is not None and g["3m"] <= -threshold
          and (g["1m"] is None or g["1m"] <= threshold)]
    return sorted(ok, key=lambda t: gains[t]["3m"])


def night_combo(tags):
    delta, used = {}, []
    for tag in tags:
        d = NIGHT_SINGLES[tag]
        if all(delta.get(k, v) == v for k, v in d.items()) and any(k not in delta for k in d):
            delta.update(d)
            used.append(tag)
    return delta, used


def night_combos(gains, threshold):
    positives = night_positives(gains, threshold)
    combos = []
    for label, tags in (("c1", positives), ("c2", [t for t in positives if t != "drop0"])):
        delta, used = night_combo(tags)
        if len(used) > 1 and all(delta != d for _, d, _ in combos):
            combos.append((f"{label}.{'+'.join(used)}", delta, used))
    return combos


def unique_runs(runs, index):
    seen = {(m["group"], m["config"]["seed"]) for m in index.values()}
    out = []
    for cfg in runs:
        key = (config_group(resolve(cfg)), cfg["seed"])
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def night_combine(runs_dir, prefix="n1", threshold=0.004, top=4):
    index = night_index(runs_dir)
    gains = night_gains(index, prefix)
    combos = night_combos(gains, threshold)
    singles = night_positives(gains, threshold)[:top]
    if len(singles) < 2:
        ranked = sorted((t for t in gains if gains[t]["3m"] is not None), key=lambda t: gains[t]["3m"])
        singles += [t for t in ranked if t not in singles][:2 - len(singles)]
    ten = [(label, delta) for label, delta, _ in combos] + [("base", {})] + \
        [(tag, NIGHT_SINGLES[tag]) for tag in singles]
    runs = [night_config(f"{prefix}-{label}-10m", delta, "10m") for label, delta in ten]
    runs += [night_config(f"{prefix}-{label}-{size}", delta, size)
             for size in ("3m", "1m") for label, delta, _ in combos]
    return {"runs": unique_runs(runs, index)}


def night_best(index, prefix, size):
    rows = [m for (name, seed), m in index.items() if seed == 0 and name and
            name.startswith(prefix + "-") and name.endswith(f"-{size}") and
            name != f"{prefix}-base-{size}"]
    return min(rows, key=lambda m: m["result"]["val_bpc"]) if rows else None


def night_variant(m, name, seed=0, **changes):
    cfg = merge(m["config"], {"name": name, "seed": seed})
    for key, value in changes.items():
        set_path(cfg, key.replace("__", "."), value)
    return cfg


def night_confirm(runs_dir, prefix="n1", long_steps=24000):
    index = night_index(runs_dir)
    gains = night_gains(index, prefix)
    runs = []
    best3 = night_best(index, prefix, "3m")
    label = best3["config"]["name"][len(prefix) + 1:-3] if best3 else None
    pair = {size: [m for m in (index.get((f"{prefix}-{label}-{size}", 0)),
                               index.get((f"{prefix}-base-{size}", 0))) if m]
            for size in ("3m", "1m")}
    for seed in (1, 2):
        for size in ("3m", "1m"):
            runs += [night_variant(m, m["config"]["name"], seed) for m in pair[size]]
        if seed == 1:
            head = len(runs)
    best10 = night_best(index, prefix, "10m")
    base10 = index.get((f"{prefix}-base-10m", 0))
    ten = [m for m in (best10, base10) if m]
    runs[head:head] = [night_variant(m, m["config"]["name"], 1) for m in ten]

    nopos = gains.get("nopos", {}).get("3m")
    pos = "none" if nopos is not None and nopos <= 0.005 else None
    longs = []
    for m in pair["3m"]:
        name = m["config"]["name"]
        if pos and m["config"]["model"].get("pos") != pos:
            longs.append(night_variant(m, f"{name}+nopos", model__pos=pos))
        for T, bs in ((2048, 8), (4096, 4)):
            if T == 4096 and m is not pair["3m"][0]:
                continue
            changes = {"task__seq_len": T, "train__batch_size": bs}
            if pos:
                changes["model__pos"] = pos
            longs.append(night_variant(m, f"{name}-T{T}", **changes))
    runs += longs
    runs += [night_variant(m, m["config"]["name"], 2) for m in ten]
    runs += [night_variant(m, f"{m['config']['name']}-{long_steps // 1000}k",
                           train__steps=long_steps) for m in pair["3m"]]
    return {"runs": unique_runs(runs, index)}


def best_lr_seeds(runs_dir, prefix, base, known=None, seeds=(1, 2)):
    rows = [m for m in collect(runs_dir) if (m["config"].get("name") or "").startswith(prefix)]
    best = min(rows, key=lambda m: m["result"]["val_bpc"]) if rows else None
    if best is None or (known and known["val_bpc"] <= best["result"]["val_bpc"]):
        train = dict(base["train"], **(known or {}).get("train", {}))
    else:
        train = best["config"]["train"]
    return {"runs": [dict(base, name=f"{prefix}seed", seed=s, train=train) for s in seeds]}
