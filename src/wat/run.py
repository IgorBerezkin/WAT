import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from wat.data import build_task, resolve_task
from wat.lab import (LMModel, LSTMBackbone, TransformerBackbone, WATBackboneX,
                     make_sched, match_embed_dim, n_params)

DEFAULTS = {
    "task": {"name": "shakespeare"},
    "model": {"name": "wat", "ctx_mode": "mean", "intra": False, "n_layers": 2,
              "chunk_size": 32, "dropout": 0.1},
    "train": {"steps": 1000, "batch_size": 32, "lr": 3e-4, "weight_decay": 0.01,
              "warmup": 100, "grad_clip": 1.0, "eval_every": 200, "eval_batches": None,
              "eval_batch_size": 32, "precision": "auto"},
    "seed": 0,
}


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_path(cfg, dotted, value):
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def resolve(cfg):
    cfg = merge(DEFAULTS, cfg)
    cfg["task"] = resolve_task(cfg["task"])
    return cfg


def config_group(cfg):
    core = {k: v for k, v in cfg.items() if k not in ("seed", "name")}
    return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]


def run_name(cfg):
    model = cfg["model"]
    tag = model["name"]
    if model["name"] == "wat":
        tag += f"-{model['ctx_mode']}" + ("-intra" if model.get("intra") else "")
    prefix = cfg.get("name") or f"{cfg['task']['name']}_{tag}"
    return f"{prefix}_{config_group(cfg)}_s{cfg['seed']}"


def build_model(mcfg, vocab, max_len, seed):
    def make(embed_dim):
        kind = mcfg["name"]
        common = dict(n_layers=mcfg["n_layers"], dropout=mcfg["dropout"])
        if kind == "wat":
            backbone = WATBackboneX(vocab, embed_dim, chunk_size=mcfg["chunk_size"],
                                    max_len=max_len, ctx_mode=mcfg["ctx_mode"],
                                    intra=mcfg["intra"], **common)
        elif kind == "transformer":
            backbone = TransformerBackbone(vocab, embed_dim, max_len=max_len, **common)
        elif kind == "lstm":
            backbone = LSTMBackbone(vocab, embed_dim, **common)
        else:
            raise ValueError(f"unknown model: {kind}")
        return LMModel(backbone, vocab)

    embed_dim = mcfg.get("embed_dim")
    if embed_dim is None:
        embed_dim, _ = match_embed_dim(make, mcfg["target_params"])
    torch.manual_seed(seed)
    return make(embed_dim), embed_dim


def pick_precision(device, requested):
    if requested != "auto":
        return requested
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def autocast(device, precision):
    if precision == "fp32" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, task, split, device, precision, batch_size, max_batches=None):
    model.eval()
    nll, correct, count = 0.0, 0, 0
    for x, y in task.eval_batches(split, batch_size, max_batches):
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logits = model(x)
        logits = logits.float()
        mask = y != -100
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        count += mask.sum().item()
    model.train()
    return {"bpc": nll / max(1, count) / math.log(2), "acc": correct / max(1, count)}


def git_info():
    commit = os.environ.get("WAT_COMMIT")
    if commit:
        return {"commit": commit, "dirty": None}
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment(device):
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": device.type,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    info.update(git_info())
    return info


def atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def atomic_torch(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run(cfg, out_root, device=None, deadline=None, log=print):
    cfg = resolve(cfg)
    name = run_name(cfg)
    out = os.path.join(out_root, name)
    metrics_path = os.path.join(out, "metrics.json")
    if os.path.exists(metrics_path):
        log(f"[{name}] done, skipped")
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs(out, exist_ok=True)
    atomic_json(os.path.join(out, "config.json"), cfg)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tcfg = cfg["train"]
    task = build_task(cfg["task"])
    model, embed_dim = build_model(cfg["model"], task.vocab, task.seq_len, cfg["seed"])
    model.to(device)
    precision = pick_precision(device, tcfg["precision"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = make_sched(opt, tcfg["steps"], tcfg["warmup"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    state = {"step": 0, "best": None, "curve": [], "train_time": 0.0}

    ckpt_path = os.path.join(out, "ckpt.pt")
    best_path = os.path.join(out, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        torch.set_rng_state(ckpt["rng"])
        if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"], device)
        log(f"[{name}] resumed at step {state['step']}")

    params = n_params(model)
    log(f"[{name}] params={params:,} embed_dim={embed_dim} device={device} precision={precision}")
    generator = torch.Generator()
    running, n_running = 0.0, 0
    model.train()
    while state["step"] < tcfg["steps"]:
        t_step = time.time()
        generator.manual_seed(cfg["seed"] * 1_000_003 + state["step"])
        x, y = task.train_batch(tcfg["batch_size"], generator)
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
                                   ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        n_running += 1
        state["step"] += 1
        state["train_time"] += time.time() - t_step

        if state["step"] % tcfg["eval_every"] == 0 or state["step"] == tcfg["steps"]:
            val = evaluate(model, task, "val", device, precision, tcfg["eval_batch_size"],
                           tcfg["eval_batches"])
            point = {"step": state["step"], "train_loss": running / n_running,
                     "val_bpc": val["bpc"], "val_acc": val["acc"],
                     "time_s": round(state["train_time"], 1)}
            state["curve"].append(point)
            running, n_running = 0.0, 0
            improved = state["best"] is None or val["bpc"] < state["best"]["val_bpc"]
            if improved:
                state["best"] = {"step": state["step"], "val_bpc": val["bpc"], "val_acc": val["acc"]}
                atomic_torch(best_path, model.state_dict())
            log(f"[{name}] step {state['step']}/{tcfg['steps']} loss={point['train_loss']:.4f} "
                f"val_bpc={val['bpc']:.4f} val_acc={val['acc'] * 100:.2f}%"
                f"{' *' if improved else ''} ({point['time_s']}s)")
            atomic_torch(ckpt_path, {
                "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "state": state,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            })
            if deadline is not None and time.time() > deadline and state["step"] < tcfg["steps"]:
                log(f"[{name}] time limit reached, checkpoint saved at step {state['step']}")
                return {"status": "interrupted", "name": name, "step": state["step"]}

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val = evaluate(model, task, "val", device, precision, tcfg["eval_batch_size"])
    test = evaluate(model, task, "test", device, precision, tcfg["eval_batch_size"])
    metrics = {
        "status": "done", "name": name, "group": config_group(cfg), "config": cfg,
        "params": params, "embed_dim": embed_dim,
        "result": {"val_bpc": val["bpc"], "val_acc": val["acc"],
                   "test_bpc": test["bpc"], "test_acc": test["acc"],
                   "best_step": state["best"]["step"]},
        "train_time_s": round(state["train_time"], 1),
        "tokens_seen": state["step"] * tcfg["batch_size"] * task.seq_len,
        "curve": state["curve"], "env": environment(device),
    }
    atomic_json(metrics_path, metrics)
    for path in (ckpt_path, best_path):
        if os.path.exists(path):
            os.remove(path)
    log(f"[{name}] test_bpc={test['bpc']:.4f} test_acc={test['acc'] * 100:.2f}%")
    return metrics


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one WAT experiment from a JSON config.")
    parser.add_argument("config")
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    for item in args.set:
        key, value = item.split("=", 1)
        set_path(cfg, key, parse_value(value))
    result = run(cfg, args.out, device=args.device)
    return 0 if result.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
