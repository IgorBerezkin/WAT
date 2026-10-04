import argparse
import copy
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from wat import speed
from wat.kernels import beam
from wat.run import atomic_json, build_model, environment

VARIANTS = {"ref": {}, "beam": {"fused_beam": True}, "read": {"fused_read": True}, "both": {"fused_beam": True, "fused_read": True}}


def error_text():
    return traceback.format_exc(limit=4)[-1500:]


def read_module():
    try:
        from wat.kernels import read
        return read
    except Exception:
        return None


def model_for(ctx, variant, size, T):
    cfg = dict(speed.model_cfg("main", size), **VARIANTS[variant])
    model, width = build_model(cfg, speed.VOCAB, T, 0)
    return model.to(ctx.device), width


def steps(ctx, variants):
    records = []
    for T in (512, 2048):
        for variant in variants:
            rec = {"variant": variant, "T": T}
            try:
                model, width = model_for(ctx, variant, "1m", T)
                x, y = ctx.batch(T)
                opt, scaler = speed.trainer(ctx, model)
                torch.cuda.reset_peak_memory_stats(ctx.device)
                rec["step"] = speed.timed(ctx, lambda: speed.train_step(ctx, model, opt, scaler, x, y))
                rec["peak_mb"] = round(torch.cuda.max_memory_allocated(ctx.device) / 2 ** 20, 1)
                rec["D"] = width
                del model, opt, scaler
            except Exception:
                rec["error"] = error_text()
            torch.cuda.empty_cache()
            records.append(rec)
            print(f"[steps] T={T} {variant}: {rec.get('step', {}).get('median_ms')} ms{' ERROR ' + rec['error'][-300:] if 'error' in rec else ''}", flush=True)
        for variant in ("ref", variants[-1]):
            rec = {"variant": variant + "+graph", "T": T}
            try:
                model, width = model_for(ctx, variant, "1m", T)
                x, y = ctx.batch(T)
                replay = speed.graph_step(ctx, model, x, y)
                rec["step"] = speed.timed(ctx, replay)
                del model, replay
            except Exception:
                rec["error"] = error_text()
            torch.cuda.empty_cache()
            records.append(rec)
            print(f"[steps] T={T} {rec['variant']}: {rec.get('step', {}).get('median_ms')} ms{' ERROR ' + rec['error'][-300:] if 'error' in rec else ''}", flush=True)
    return records


def trajectory(ctx, variants, n=30):
    out = {}
    for variant in variants:
        try:
            model, _ = model_for(ctx, variant, "1m", 512)
            opt, scaler = speed.trainer(ctx, model)
            losses = []
            for i in range(n):
                x, y = ctx.batch(512, seed=100 + i)
                losses.append(round(speed.train_step(ctx, model, opt, scaler, x, y), 5))
            out[variant] = losses
            del model, opt, scaler
        except Exception:
            out[variant] = {"error": error_text()}
        torch.cuda.empty_cache()
    ref = out.get("ref")
    if isinstance(ref, list):
        out["max_abs_diff"] = {v: round(max(abs(a - b) for a, b in zip(ref, l)), 5) for v, l in out.items() if v != "ref" and isinstance(l, list)}
    return out


def passed(data, key):
    records = data.get(key)
    return isinstance(records, list) and bool(records) and all(isinstance(r, dict) and r.get("pass") for r in records)


def plan(runs_dir, runs=None, check=None):
    root = os.path.dirname(os.path.abspath(runs_dir))
    data = {}
    if check:
        try:
            with open(os.path.join(root, check), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        if not passed(data, "beam_check"):
            print("beam kernel check failed: training stage skipped", flush=True)
            return {"runs": []}
    out = []
    for run in runs or []:
        run = copy.deepcopy(run)
        if run["model"].get("fused_read") and check and not passed(data, "read_check"):
            run["model"].pop("fused_read")
            run["name"] += "-noread"
        out.append(run)
    print(f"training stage: {[r['name'] for r in out]}", flush=True)
    return {"runs": out}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check and time the fused WAT 1.0 kernels on a GPU.")
    parser.add_argument("--out", default=os.path.join("results", "kernels"))
    parser.add_argument("--minutes", type=float, default=30.0)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    ctx = speed.Ctx(SimpleNamespace(device="cuda", quick=False, minutes=args.minutes, out=args.out))
    path = os.path.join(args.out, "kernels.json")
    report = {"env": environment(ctx.device), "beam_available": beam.available(), "t0": round(time.time(), 1)}
    read = read_module()
    report["read_available"] = bool(read and read.available())

    def save():
        atomic_json(path, report)

    save()
    report["beam_check"] = beam.check()
    save()
    print(f"[beam check] {json.dumps(report['beam_check'])[:2000]}", flush=True)
    report["beam_bench"] = beam.bench()
    save()
    print(f"[beam bench] {json.dumps(report['beam_bench'])[:2000]}", flush=True)
    report["beam_bench_bt16"] = beam.bench(block_t=16)
    save()
    print(f"[beam bench bt16] {json.dumps(report['beam_bench_bt16'])[:2000]}", flush=True)
    variants = ["ref", "beam"]
    if report["read_available"]:
        try:
            report["read_check"] = read.check()
        except Exception:
            report["read_check"] = {"error": error_text()}
        save()
        print(f"[read check] {json.dumps(report['read_check'])[:2000]}", flush=True)
        try:
            report["read_bench"] = read.bench()
        except Exception:
            report["read_bench"] = {"error": error_text()}
        save()
        print(f"[read bench] {json.dumps(report['read_bench'])[:2000]}", flush=True)
        variants += ["read", "both"]
    report["trajectory"] = trajectory(ctx, variants)
    save()
    print(f"[trajectory] {json.dumps(report['trajectory'].get('max_abs_diff'))}", flush=True)
    report["steps"] = steps(ctx, variants)
    report["t1"] = round(time.time(), 1)
    save()
    print("=== kernel check finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
