import argparse
import contextlib
import copy
import faulthandler
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from wat.baselines.transformer import CausalSelfAttention, SwiGLU, TransformerBackbone
from wat.common import LMModel, RMSNorm, match_pointers, n_params
from wat.main.model import MainBackbone, MainBlock, TreeSearch
from wat.run import atomic_json, build_model, environment, pick_precision

VOCAB = 256
PTR = (4, 8, 16, 32)
SIZES = {"1m": (1_000_000, 3), "3m": (3_000_000, 4), "10m": (10_000_000, 6)}
WIDTHS = {"1m": 104, "3m": 168, "10m": 256}
TF_WIDTH = 160
PARTS = ("P0_pointer", "P1_filter", "P2_pad", "P3_tree", "P4_read", "P5_inject", "P6_search", "P6a_proj",
         "P6b_tree", "P6c_beam", "P6d_out", "P7_ffn", "P8_embed", "P8_head", "P8_opt", "T_attn", "T_ffn")
GROUPS = {"P6": ("P6_search", "P6a_proj", "P6b_tree", "P6c_beam", "P6d_out"), "P8": ("P8_embed", "P8_head", "P8_opt"),
          "T": ("T_attn", "T_ffn")}
MICRO = ("P0_pointer", "P1_filter", "P3_tree", "P4_read", "P5_inject", "P6_search", "P6b_tree", "P6c_beam",
         "P7_ffn", "P8_head", "T_attn", "T_block", "M_block", "C_block")
TAG = "FWD"
CLASSES = (("attention", r"fmha|flash|attention|sdpa|efficient"), ("conv", r"conv|cudnn|winograd|implicit"),
           ("gemm", r"gemm|cutlass|cublas|matmul|xmma|s1688|h1688|s884|h884"), ("sort", r"sort|radix|topk|bitonic"),
           ("softmax", r"softmax"), ("reduce", r"reduce"), ("gather", r"gather|scatter|index|embedding"),
           ("elementwise", r"elementwise|vectorized|unrolled|copy|fill|cat"))
BUCKETS = ((5, "<5us"), (20, "5-20us"), (100, "20-100us"), (float("inf"), ">100us"))
SDPA = {-1: "error", 0: "math", 1: "flash", 2: "efficient", 3: "cudnn", 4: "overrideable"}
DATASHEET = {"copy_gbps": 320.0, "fp16_tflops": 65.0}
SEMANTICS = {
    "plain.median_ms / step.median_ms": "median of synchronized full steps, no profiler",
    "fwd_ms / fwdbwd_ms / device_ms": "pipelined time per call without per-iteration sync, median of 3 samples",
    "kernel_ms / kernel_count": "per step and exclusive: a forward kernel belongs to the innermost part range, a backward kernel to the part of its autograd node",
    "kernel_groups_ms": "kernel_ms with part groups summed: P6 = P6_search + P6a..P6d, P8 = embed + head + opt, T = attn + ffn",
    "cpu.op_ms / api_ms / wait_ms / python_ms": "per step host self time: aten ops, CUDA API calls (launches), synchronization waits, Python between ops inside part ranges",
    "fwd_cpu_incl_ms": "per step inclusive forward range durations; nested (P6_search contains P6a..P6d, P6c_beam contains P6b_tree)",
    "bwd_cpu_incl_ms": "per step inclusive autograd node durations grouped by forward part",
    "gpu_busy_unprofiled": "gpu kernel ms of a profiled step divided by the unprofiled median step",
    "micro P6c_beam": "TreeSearch._beam including its _tree call; in test A P6c_beam excludes P6b_tree",
    "fused_min_bytes": "memory traffic lower bound of one fused forward kernel; impl_bytes_est is the rough traffic as implemented",
}


class Ctx:
    def __init__(self, args):
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cuda = self.device.type == "cuda"
        self.precision = pick_precision(self.device, "auto")
        self.quick = args.quick
        self.tokens = 1024 if args.quick else 16384
        self.lengths = (128, 256) if args.quick else (512, 2048, 8192)
        self.warmup, self.iters = (1, 2) if args.quick else (5, 20)
        self.prof_steps = 1 if args.quick else 3
        self.deadline = time.time() + args.minutes * 60
        self.out = args.out
        self.calib = {}
        self.compiled_once = False
        if self.cuda:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
                torch.ones(8, device=self.device).add_(1)
                torch.cuda.synchronize(self.device)

    def amp(self, cache=True):
        if self.precision == "fp32" or not self.cuda:
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype, cache_enabled=cache)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def left(self):
        return self.deadline - time.time()

    def batch(self, T, seed=0):
        g = torch.Generator().manual_seed(seed)
        B = max(1, self.tokens // T)
        x = torch.randint(0, VOCAB, (B, T), generator=g).to(self.device)
        y = torch.randint(0, VOCAB, (B, T), generator=g).to(self.device)
        return x, y

    def free(self):
        if self.cuda:
            torch.cuda.empty_cache()


class Clocks:
    QUERY = "index,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"

    def __init__(self):
        self.samples, self.proc, self.thread = [], None, None
        try:
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits",
                                          "-lms", "500"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1)
        except OSError:
            return
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(self.QUERY.split(",")):
                self.samples.append([round(time.time(), 2)] + parts)

    def snapshot(self):
        return {"columns": ["t"] + self.QUERY.split(","), "samples": list(self.samples)}

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(2)
        return self.snapshot()


def model_cfg(kind, size):
    target, layers = SIZES[size]
    base = {"target_params": target, "n_layers": layers, "dropout": 0.0}
    if kind == "main":
        return dict(base, name="wat_main", mem="beam8", ptr=list(PTR))
    if kind == "combo":
        return dict(base, name="wat_main", mem=None, ptr=None)
    return dict(base, name="transformer", pos="rope", ffn="swiglu", attn="sdpa")


def make_model(ctx, kind, size, T):
    model, width = build_model(model_cfg(kind, size), VOCAB, T, 0)
    return model.to(ctx.device), width


def tf_heads(D):
    heads = 1
    for h in (1, 2, 4):
        if D % h == 0 and D // h >= 8:
            heads = h
    return heads


def aligned_heads(D):
    good = [h for h in (4, 2, 1) if D % h == 0 and D // h >= 8 and (D // h) % 8 == 0]
    return good[0] if good else tf_heads(D)


def sdpa_backend(ctx, B, heads, T, head_dim):
    if not ctx.cuda:
        return None
    try:
        q = torch.randn(B, heads, T, head_dim, device=ctx.device, dtype=torch.float16)
        return SDPA.get(int(torch._fused_sdp_choice(q, q, q, is_causal=True)), "unknown")
    except Exception:
        return "probe_failed"


def ranged(fn, name):
    def inner(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)
    return inner


def block_forward(self, x):
    B, T, D = x.shape
    with record_function("P1_filter"):
        h = self.conv(self.norm_conv(x))
        x = x + h * torch.sigmoid(self.W_gate(h))
    with record_function("P2_pad"):
        K = 1 << (T - 1).bit_length()
        x_padded = x if K == T else torch.cat([x, x[:, -1:, :].expand(-1, K - T, -1)], dim=1)
        grid = x_padded.view(B, 1, K, D)
    with record_function("P3_tree"):
        levels = self.tree(grid)
    with record_function("P4_read"):
        read = self.read(levels, grid)
    with record_function("P5_inject"):
        ctx = self.ctx_norm_m(read.reshape(B, K, D))
        h_ctx = (x_padded + self.W_global(ctx))[:, :T, :]
        x = x + (h_ctx - x.detach()) * 0.5
    if self.memory is not None:
        with record_function("P6_search"):
            x = x + self.memory(x)
    with record_function("P7_ffn"):
        return x + self.ffn(self.norm_ffn(x))


def search_forward(self, x):
    B, T, D = x.shape
    with record_function("P6a_proj"):
        h = self.norm(x)
        q, k, u = self.split(self.W_q(h)), self.split(self.W_k(h)), self.split(self.W_u(h))
        u_next = torch.cat([u[:, :, 1:], torch.zeros_like(u[:, :, :1])], dim=2)
    with torch.autocast("cuda", enabled=False):
        out = self.read(h, q, k, u_next)
    with record_function("P6d_out"):
        return self.W_o(out.transpose(1, 2).reshape(B, T, D).to(x.dtype))


def backbone_forward(self, x):
    with record_function("P8_embed"):
        h = self.embedding(x)
    with record_function("P0_pointer"):
        cands = match_pointers(x, self.ptr, self.vocab)
    with record_function("P8_embed"):
        for emb, cand in zip(self.ptr_emb, cands):
            h = h + emb(cand)
        h = self.input_dropout(h)
    for layer in self.layers:
        h = self.layer_dropout(layer(h))
    with record_function("P8_head"):
        return self.output_norm(h)


def tf_forward(self, x):
    with record_function("P8_embed"):
        h = self.embedding(x)
        if self.pos_encoding is not None:
            h = h + self.pos_encoding(torch.arange(x.size(1), device=x.device))
    with record_function("P0_pointer"):
        cands = match_pointers(x, self.ptr, self.vocab)
    with record_function("P8_embed"):
        for emb, cand in zip(self.ptr_emb, cands):
            h = h + emb(cand)
        h = self.input_dropout(h)
    for b in self.layers:
        with record_function("T_attn"):
            h = h + b.drop(b.attn(b.norm1(h)))
        with record_function("T_ffn"):
            h = h + b.drop(b.ffn(b.norm2(h)))
    with record_function("P8_head"):
        return self.output_norm(h)


def lm_forward(self, x):
    h = self.backbone(x)
    with record_function("P8_head"):
        return self.head(h)


@contextlib.contextmanager
def instrumented():
    patches = [(MainBlock, "forward", block_forward), (TreeSearch, "forward", search_forward),
               (MainBackbone, "forward", backbone_forward), (TransformerBackbone, "forward", tf_forward),
               (LMModel, "forward", lm_forward),
               (TreeSearch, "_tree", ranged(TreeSearch._tree, "P6b_tree")),
               (TreeSearch, "_beam", ranged(TreeSearch._beam, "P6c_beam"))]
    saved = [(cls, name, cls.__dict__[name]) for cls, name, _ in patches]
    try:
        for cls, name, fn in patches:
            setattr(cls, name, fn)
        yield
    finally:
        for cls, name, fn in saved:
            setattr(cls, name, fn)


def train_step(ctx, model, opt, scaler, x, y, marks=False):
    rf = record_function if marks else (lambda name: contextlib.nullcontext())
    with ctx.amp():
        logits = model(x)
        with rf("P8_head"):
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=-100)
    with rf("P8_opt"):
        opt.zero_grad(set_to_none=True)
    with rf("BWD"):
        scaler.scale(loss).backward()
    with rf("P8_opt"):
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    return loss.item()


def trainer(ctx, model, capturable=False):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, capturable=capturable)
    scaler = torch.amp.GradScaler("cuda", enabled=ctx.cuda and ctx.precision == "fp16")
    return opt, scaler


def timed(ctx, fn, warmup=None, iters=None):
    warmup = ctx.warmup if warmup is None else warmup
    iters = ctx.iters if iters is None else iters
    for _ in range(warmup):
        fn()
    ctx.sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ctx.sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": round(statistics.median(times), 3), "min_ms": round(min(times), 3),
            "mean_ms": round(statistics.fmean(times), 3), "iters": iters}


def throughput(ctx, fn, warmup=None, iters=None, samples=3):
    warmup = ctx.warmup if warmup is None else warmup
    iters = ctx.iters if iters is None else iters
    for _ in range(warmup):
        fn()
    ctx.sync()

    def sample(n):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ctx.sync()
        return time.perf_counter() - t0

    total = sample(iters)
    while not ctx.quick and total < 0.3 and iters < 2000:
        iters = min(2000, max(iters * 2, int(iters * 0.35 / max(total, 1e-4))))
        total = sample(iters)
    values = [total] + [sample(iters) for _ in range(0 if ctx.quick else samples - 1)]
    return round(statistics.median(values) * 1e3 / iters, 4)


def kernel_class(name, cat):
    if cat != "kernel":
        return "memcpy"
    low = name.lower()
    for label, pattern in CLASSES:
        if re.search(pattern, low):
            return label
    return "other"


def grouped(values):
    out = defaultdict(float)
    for key, v in values.items():
        phase, part = key.split(":", 1)
        group = next((g for g, members in GROUPS.items() if part in members), part)
        out[f"{phase}:{group}"] += v
    return out


def analyze_trace(path, steps):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    host = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver")]
    device = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    parent, inner = {}, defaultdict(float)
    by_thread = defaultdict(list)
    for e in host:
        by_thread[(e.get("pid"), e.get("tid"))].append(e)
    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        stack = []
        for e in evs:
            while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) <= e["ts"]:
                stack.pop()
            parent[id(e)] = stack[-1] if stack else None
            if stack:
                inner[id(stack[-1])] += e.get("dur", 0)
            stack.append(e)

    def chain(e):
        while e is not None:
            yield e
            e = parent[id(e)]

    def tag_of(a):
        if a.get("cat") != "user_annotation":
            return None
        name = a.get("name", "")
        return name if name in PARTS else ("part" if name == TAG else None)

    def forward_part(e):
        for a in chain(e):
            name = a.get("name", "")
            if name.startswith("autograd::engine::evaluate_function") or name == "BWD":
                return None
            tag = tag_of(a)
            if tag:
                return tag
        return None

    seq_part = {}
    for e in sorted((e for e in host if e.get("cat") == "cpu_op"), key=lambda e: e["ts"]):
        seq = e.get("args", {}).get("Sequence number")
        if seq is not None:
            part = forward_part(e)
            if part is not None:
                seq_part[seq] = part

    memo = {}

    def place(e):
        key = id(e)
        if key in memo:
            return memo[key]
        result = ("outside", "other")
        for a in chain(e):
            name = a.get("name", "")
            if name.startswith("autograd::engine::evaluate_function"):
                seq = a.get("args", {}).get("Sequence number")
                result = ("bwd", seq_part.get(seq, "bwd_other") if seq is not None else "grad_accum")
                break
            if name == "BWD":
                result = ("bwd", "bwd_other")
                break
            tag = tag_of(a)
            if tag:
                result = ("opt", tag) if tag == "P8_opt" else ("fwd", tag)
                break
            if a.get("cat") == "user_annotation" and name in ("STEP", "PART"):
                result = ("fwd", "other")
                break
        memo[key] = result
        return result

    ext, corr = {}, {}
    for e in host:
        args = e.get("args", {})
        if e.get("cat") == "cpu_op" and "External id" in args:
            ext[args["External id"]] = e
        if "correlation" in args:
            corr[args["correlation"]] = e
    kernel_ms, kernel_n, class_ms = defaultdict(float), defaultdict(int), defaultdict(float)
    hist_n, hist_ms = defaultdict(int), defaultdict(float)
    names = defaultdict(lambda: [0.0, 0])
    by_part = defaultdict(lambda: defaultdict(float))
    spans = []
    for e in device:
        args = e.get("args", {})
        dur_us = e.get("dur", 0)
        dur = dur_us / 1e3
        src = ext.get(args.get("External id")) or corr.get(args.get("correlation"))
        phase, part = place(src) if src is not None else ("unknown", "other")
        key = f"{phase}:{part}"
        kernel_ms[key] += dur
        kernel_n[key] += 1
        class_ms[kernel_class(e.get("name", ""), e.get("cat"))] += dur
        if e.get("cat") == "kernel":
            label = next(lbl for limit, lbl in BUCKETS if dur_us < limit)
            hist_n[label] += 1
            hist_ms[label] += dur
        short = e.get("name", "")[:90]
        names[short][0] += dur
        names[short][1] += 1
        by_part[key][short] += dur
        spans.append((e["ts"], e["ts"] + dur_us))
    cpu = {"op_ms": defaultdict(float), "api_ms": defaultdict(float), "wait_ms": defaultdict(float),
           "python_ms": defaultdict(float)}
    fwd_incl, bwd_incl = defaultdict(float), defaultdict(float)
    step_wall = []
    for e in host:
        cat, name = e.get("cat"), e.get("name", "")
        own = max(0.0, e.get("dur", 0) - inner[id(e)]) / 1e3
        if cat == "cpu_op":
            phase, part = place(e)
            cpu["op_ms"][f"{phase}:{part}"] += own
            if name.startswith("autograd::engine::evaluate_function"):
                seq = e.get("args", {}).get("Sequence number")
                bwd_incl[seq_part.get(seq, "bwd_other") if seq is not None else "grad_accum"] += e.get("dur", 0) / 1e3
        elif cat in ("cuda_runtime", "cuda_driver"):
            phase, part = place(e)
            bucket = "wait_ms" if re.search(r"Synchronize|cudaMemcpy(?!Async)|cuMemcpyDtoH(?!Async)", name) else "api_ms"
            cpu[bucket][f"{phase}:{part}"] += own
        elif cat == "user_annotation":
            if name == "STEP":
                step_wall.append(e.get("dur", 0) / 1e3)
            tag = tag_of(e)
            if tag is not None:
                phase, part = place(e)
                cpu["python_ms"][f"{phase}:{part}"] += own
                fwd_incl[tag] += e.get("dur", 0) / 1e3
            elif name == "BWD":
                fwd_incl["BWD"] += e.get("dur", 0) / 1e3
    busy, span = 0.0, 0.0
    if spans:
        spans.sort()
        start, end = spans[0]
        first, last = spans[0][0], max(s[1] for s in spans)
        for a, b in spans[1:]:
            if a > end:
                busy += end - start
                start, end = a, b
            else:
                end = max(end, b)
        busy += end - start
        span = last - first
    n = max(1, steps)
    per = lambda d: {k: round(v / n, 4) for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
    kernels = sum(1 for e in device if e.get("cat") == "kernel")
    total = sum(kernel_ms.values())
    lost = sum(v for k, v in kernel_ms.items() if k.startswith(("outside", "unknown")))
    return {
        "steps": steps,
        "kernel_ms": per(kernel_ms),
        "kernel_count": {k: round(v / n, 1) for k, v in sorted(kernel_n.items(), key=lambda kv: -kv[1])},
        "kernel_groups_ms": per(grouped(kernel_ms)),
        "cpu": {k: per(v) for k, v in cpu.items()},
        "cpu_groups": {k: per(grouped(v)) for k, v in cpu.items()},
        "fwd_cpu_incl_ms": per(fwd_incl), "bwd_cpu_incl_ms": per(bwd_incl),
        "class_ms": per(class_ms),
        "kernel_hist": {lbl: {"count": round(hist_n[lbl] / n, 1), "ms": round(hist_ms[lbl] / n, 4)} for _, lbl in BUCKETS},
        "kernels_per_step": round(kernels / n, 1),
        "gpu_kernel_ms_per_step": round(total / n, 4),
        "gpu_span_ms_per_step": round(span / 1e3 / n, 4),
        "gpu_busy_fraction": round(busy / span, 4) if span else None,
        "step_cpu_wall_ms": round(statistics.fmean(step_wall), 3) if step_wall else None,
        "mean_kernel_us": round(1e3 * total / max(1, kernels), 2),
        "unattributed_fraction": round(lost / total, 4) if total else None,
        "top_kernels": [{"name": k, "ms": round(v[0] / n, 4), "count": round(v[1] / n, 1)}
                        for k, v in sorted(names.items(), key=lambda kv: -kv[1][0])[:15]],
        "top_kernels_by_part": {key: [{"name": k, "ms": round(v / n, 4)} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:3]]
                                for key, d in sorted(by_part.items())},
    }


def profile_run(ctx, fn, steps, label="STEP"):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if ctx.cuda else [])
    try:
        with profile(activities=activities, schedule=schedule(wait=0, warmup=1, active=steps, repeat=1),
                     on_trace_ready=lambda p: p.export_chrome_trace(path), record_shapes=False, with_stack=False) as prof:
            for _ in range(steps + 1):
                with record_function(label):
                    fn()
                ctx.sync()
                prof.step()
        return analyze_trace(path, steps)
    finally:
        os.remove(path)


def save(ctx, name, payload):
    atomic_json(os.path.join(ctx.out, f"speed_{name}.json"), payload)


def error_text():
    return traceback.format_exc(limit=4)[-1500:]


def count_skipped(node):
    if isinstance(node, dict):
        return int("skipped" in node) + sum(count_skipped(v) for v in node.values() if isinstance(v, (dict, list)))
    if isinstance(node, list):
        return sum(count_skipped(v) for v in node)
    return 0


def same_output(ctx, model, x):
    model.eval()
    with torch.no_grad(), ctx.amp():
        plain = model(x).float()
        with instrumented():
            marked = model(x).float()
    model.train()
    return float((plain - marked).abs().max())


def test_a(ctx):
    sizes = ("1m",) if ctx.quick else ("1m", "3m")
    records = []
    for kind in ("tfm", "combo", "main"):
        for size in sizes:
            for T in ctx.lengths[:2]:
                rec = {"kind": kind, "size": size, "T": T, "t": [round(time.time(), 2)]}
                records.append(rec)
                if ctx.left() < 60:
                    rec["skipped"] = "time"
                    save(ctx, "A", {"semantics": SEMANTICS, "records": records})
                    continue
                try:
                    model, width = make_model(ctx, kind, size, T)
                    x, y = ctx.batch(T)
                    rec.update(B=x.size(0), D=width, params=n_params(model))
                    if kind == "tfm":
                        heads = tf_heads(width)
                        rec.update(heads=heads, head_dim=width // heads,
                                   sdpa_backend=sdpa_backend(ctx, x.size(0), heads, T, width // heads))
                    opt, scaler = trainer(ctx, model)
                    if ctx.cuda:
                        torch.cuda.reset_peak_memory_stats(ctx.device)
                    rec["plain"] = timed(ctx, lambda: train_step(ctx, model, opt, scaler, x, y))
                    if ctx.cuda:
                        rec["peak_mb"] = round(torch.cuda.max_memory_allocated(ctx.device) / 2 ** 20, 1)
                    rec["instrument_max_diff"] = same_output(ctx, model, x)
                    with instrumented():
                        for _ in range(2):
                            train_step(ctx, model, opt, scaler, x, y, marks=True)
                        ctx.sync()
                        rec["profile"] = profile_run(ctx, lambda: train_step(ctx, model, opt, scaler, x, y, marks=True), ctx.prof_steps)
                    rec["gpu_busy_unprofiled"] = round(rec["profile"]["gpu_kernel_ms_per_step"] / rec["plain"]["median_ms"], 4)
                    del model, opt, scaler
                except Exception:
                    rec["error"] = error_text()
                rec["t"].append(round(time.time(), 2))
                ctx.free()
                save(ctx, "A", {"semantics": SEMANTICS, "records": records})
                prof = rec.get("profile", {})
                print(f"[A] {kind} {size} T={T}: {rec.get('plain', {}).get('median_ms')} ms, kernels {prof.get('gpu_kernel_ms_per_step')} ms, "
                      f"busy {rec.get('gpu_busy_unprofiled')}, unattributed {prof.get('unattributed_fraction')}"
                      f"{' ERROR ' + rec['error'][-300:] if 'error' in rec else ''}", flush=True)
    return records


def part_spec(name, D, B, T, device):
    K = 1 << (T - 1).bit_length()
    g = torch.Generator().manual_seed(0)

    def leaf(*shape, dtype=torch.float32):
        return (torch.randn(*shape, generator=g) * 0.5).to(device=device, dtype=dtype).requires_grad_()

    if name == "P0_pointer":
        x = torch.randint(0, VOCAB, (B, T), generator=g).to(device)
        return {"fn": lambda x: match_pointers(x, list(PTR), VOCAB), "inputs": [x], "modules": [], "backward": False}
    if name in ("M_block", "C_block"):
        block = MainBlock(D, max_len=T, mem="beam8" if name == "M_block" else None).to(device)
        return {"fn": block, "inputs": [leaf(B, T, D)], "modules": [block]}
    if name in ("P6_search", "P6b_tree", "P6c_beam"):
        mem = TreeSearch(D, "beam8").to(device)
        if name == "P6_search":
            return {"fn": mem, "inputs": [leaf(B, T, D)], "modules": [mem]}
        half = torch.float16 if device.type == "cuda" else torch.float32
        qku = [leaf(B, 4, T, D // 4, dtype=half) for _ in range(3)]
        fn = (lambda q, k, u: mem._tree(q, k, u)[-1]) if name == "P6b_tree" else (lambda q, k, u: mem._beam(q, k, u, 8))
        return {"fn": fn, "inputs": qku, "modules": [mem], "autocast": False}
    if name == "P8_head":
        head = nn.Linear(D, VOCAB).to(device)
        y = torch.randint(0, VOCAB, (B * T,), generator=g).to(device)
        return {"fn": lambda x: F.cross_entropy(head(x).reshape(-1, VOCAB).float(), y), "inputs": [leaf(B, T, D)],
                "modules": [head], "scalar": True}
    if name in ("T_attn", "T_block"):
        heads = aligned_heads(D)
        norm1, attn = RMSNorm(D).to(device), CausalSelfAttention(D, heads, 0.0, T, rope=True, sdpa=True).to(device)
        norm2, ffn = RMSNorm(D).to(device), SwiGLU(D).to(device)
        if name == "T_attn":
            return {"fn": lambda x: x + attn(norm1(x)), "inputs": [leaf(B, T, D)], "modules": [norm1, attn], "heads": heads}

        def tblock(x):
            x = x + attn(norm1(x))
            return x + ffn(norm2(x))
        return {"fn": tblock, "inputs": [leaf(B, T, D)], "modules": [norm1, attn, norm2, ffn], "heads": heads}
    block = MainBlock(D, max_len=T, mem=None).to(device)
    if name == "P1_filter":
        def filt(x):
            h = block.conv(block.norm_conv(x))
            return x + h * torch.sigmoid(block.W_gate(h))
        return {"fn": filt, "inputs": [leaf(B, T, D)], "modules": [block]}
    if name == "P3_tree":
        return {"fn": lambda grid: block.tree(grid)[1:], "inputs": [leaf(B, 1, K, D)], "modules": [block]}
    if name == "P4_read":
        with torch.no_grad():
            levels = block.tree(torch.randn(B, 1, K, D, generator=g).to(device))
        levels = [lv.detach().requires_grad_() for lv in levels]
        return {"fn": lambda *lv: block.read(list(lv), lv[0]), "inputs": levels, "modules": [block]}
    if name == "P5_inject":
        def inject(xp, r):
            x = xp[:, :T]
            ctx_ = block.ctx_norm_m(r.reshape(B, K, D))
            h_ctx = (xp + block.W_global(ctx_))[:, :T, :]
            return x + (h_ctx - x.detach()) * 0.5
        return {"fn": inject, "inputs": [leaf(B, K, D), leaf(B, 1, K, D)], "modules": [block]}
    if name == "P7_ffn":
        return {"fn": lambda x: x + block.ffn(block.norm_ffn(x)), "inputs": [leaf(B, T, D)], "modules": [block]}
    raise ValueError(name)


def gemm_flops(name, D, B, T):
    K = 1 << (T - 1).bit_length()
    hidden = max(8, int(8 * D / 3) // 8 * 8)
    base = {"P1_filter": 8 * D * D * B * T, "P3_tree": 12 * D * D * B * K, "P4_read": 2 * D * D * B * K,
            "P5_inject": 2 * D * D * B * K, "P6_search": 8 * D * D * B * T, "P7_ffn": 16 * D * D * B * T,
            "P8_head": 2 * D * VOCAB * B * T, "T_attn": 8 * D * D * B * T + 2 * B * T * T * D}
    base["T_block"] = base["T_attn"] + 6 * D * hidden * B * T
    base["C_block"] = sum(base[k] for k in ("P1_filter", "P3_tree", "P4_read", "P5_inject", "P7_ffn"))
    base["M_block"] = base["C_block"] + base["P6_search"]
    return base.get(name)


def traffic(name, D, B, T):
    K = 1 << (T - 1).bit_length()
    L = max(1, K.bit_length() - 1)
    bkd, btd = B * K * D, B * T * D
    fused = {"P1_filter": 8 * btd, "P3_tree": 8 * bkd, "P4_read": 16 * bkd, "P5_inject": 8 * bkd + 4 * btd,
             "P6c_beam": 8 * btd, "P7_ffn": 8 * btd}
    impl = {"P4_read": 50 * bkd * L, "P6c_beam": 8 * btd + 18 * (L + 1) * btd + (L + 1) * B * 4 * T * 17 * 12 * 2}
    return fused.get(name), impl.get(name)


def run_part(ctx, name, D, T):
    B = max(1, ctx.tokens // T)
    spec = part_spec(name, D, B, T, ctx.device)
    fn, inputs = spec["fn"], spec["inputs"]
    params = [p for m in spec["modules"] for p in m.parameters()]
    leaves = [t for t in inputs if t.requires_grad] + params
    backward = spec.get("backward", True)
    if spec.get("autocast", True):
        amp = ctx.amp
    else:
        amp = (lambda: torch.autocast("cuda", enabled=False)) if ctx.cuda else contextlib.nullcontext

    def forward(tag=False):
        with (record_function(TAG) if tag else contextlib.nullcontext()), amp():
            out = fn(*inputs)
        return out if isinstance(out, (list, tuple)) else [out]

    rec = {"part": name, "D": D, "T": T, "B": B, "t": [round(time.time(), 2)]}
    if "heads" in spec:
        heads = spec["heads"]
        rec.update(heads=heads, head_dim=D // heads, sdpa_backend=sdpa_backend(ctx, B, heads, T, D // heads))
    grads = None
    if backward and not spec.get("scalar"):
        with torch.no_grad():
            grads = [torch.randn_like(o) for o in forward()]
    rec["fwd_ms"] = throughput(ctx, forward)

    def fwdbwd(tag=False):
        for t in leaves:
            t.grad = None
        outs = forward(tag)
        if grads is None:
            outs[0].backward()
        else:
            torch.autograd.backward(outs, grads)

    if backward:
        rec["fwdbwd_ms"] = throughput(ctx, fwdbwd)
    if ctx.cuda:
        try:
            prof = profile_run(ctx, (lambda: fwdbwd(True)) if backward else (lambda: forward(True)), 1, label="PART")
            fwd_k = prof["kernel_ms"].get("fwd:part", 0.0)
            rec.update(kernels=prof["kernels_per_step"], fwd_kernels=prof["kernel_count"].get("fwd:part", 0.0),
                       kernel_ms_total=prof["gpu_kernel_ms_per_step"], fwd_kernel_ms=round(fwd_k, 4),
                       bwd_kernel_ms=round(sum(v for k, v in prof["kernel_ms"].items() if k.startswith("bwd:")), 4),
                       class_ms=prof["class_ms"], kernel_hist=prof["kernel_hist"], mean_kernel_us=prof["mean_kernel_us"],
                       top_kernels=prof["top_kernels"][:5])
        except Exception:
            rec["profile_error"] = error_text()
    gbps = ctx.calib.get("copy_gbps") or DATASHEET["copy_gbps"]
    tflops = ctx.calib.get("fp16_tflops") or DATASHEET["fp16_tflops"]
    flops = gemm_flops(name, D, B, T)
    fused, impl = traffic(name, D, B, T)
    bounds = []
    if flops:
        rec["gemm_flops_fwd"] = flops
        rec["compute_bound_fwd_ms"] = round(flops / (tflops * 1e12) * 1e3, 5)
        bounds.append(rec["compute_bound_fwd_ms"])
    if fused:
        rec["fused_min_bytes"] = fused
        rec["memory_bound_fwd_ms"] = round(fused / (gbps * 1e9) * 1e3, 5)
        bounds.append(rec["memory_bound_fwd_ms"])
    if bounds:
        rec["roofline_fwd_ms"] = max(bounds)
    fwd_k = rec.get("fwd_kernel_ms")
    if fwd_k:
        if fused:
            rec["fwd_kernel_gbps_fused_bytes"] = round(fused / (fwd_k * 1e-3) / 1e9, 2)
        if impl:
            rec["impl_bytes_est"] = impl
            rec["fwd_kernel_gbps_impl_bytes"] = round(impl / (fwd_k * 1e-3) / 1e9, 2)
        if flops:
            rec["fwd_kernel_tflops"] = round(flops / (fwd_k * 1e-3) / 1e12, 3)
    rec["t"].append(round(time.time(), 2))
    return rec


def micro(ctx, label, cases):
    records = []
    for name, D, T in cases:
        if ctx.left() < 30:
            records.append({"part": name, "D": D, "T": T, "skipped": "time"})
            save(ctx, label, {"semantics": SEMANTICS, "tokens_per_step": ctx.tokens, "records": records})
            continue
        try:
            rec = run_part(ctx, name, D, T)
        except Exception:
            rec = {"part": name, "D": D, "T": T, "error": error_text()}
        records.append(rec)
        ctx.free()
        save(ctx, label, {"semantics": SEMANTICS, "tokens_per_step": ctx.tokens, "records": records})
        print(f"[{label}] {name} D={D} T={T}: fwd {rec.get('fwd_ms')} ms (kernels {rec.get('fwd_kernel_ms')}), "
              f"fwd+bwd {rec.get('fwdbwd_ms')} ms, kernels {rec.get('kernels')}"
              f"{' sdpa ' + str(rec['sdpa_backend']) if 'sdpa_backend' in rec else ''}"
              f"{' ERROR' if 'error' in rec else ''}{' PROFILE_ERROR' if 'profile_error' in rec else ''}", flush=True)
    return records


def tf_width(ctx):
    return 32 if ctx.quick else TF_WIDTH


def test_b(ctx):
    widths = (32,) if ctx.quick else tuple(WIDTHS.values())
    T = ctx.lengths[0]
    cases = [(name, D, T) for D in widths for name in MICRO]
    cases += [("T_attn", tf_width(ctx), T), ("T_block", tf_width(ctx), T)]
    return micro(ctx, "B", cases)


def test_c(ctx):
    D = 32 if ctx.quick else WIDTHS["1m"]
    cases = []
    for T in ctx.lengths:
        cases += [(name, D, T) for name in MICRO]
        cases += [("T_attn", tf_width(ctx), T), ("T_block", tf_width(ctx), T)]
    return micro(ctx, "C", cases)


class PtrOutside(nn.Module):
    def __init__(self, lm, cands):
        super().__init__()
        self.lm, self.cands = lm, cands

    def forward(self, x):
        bb = self.lm.backbone
        h = bb.embedding(x)
        for emb, cand in zip(bb.ptr_emb, self.cands):
            h = h + emb(cand)
        h = bb.input_dropout(h)
        for layer in bb.layers:
            h = bb.layer_dropout(layer(h))
        return self.lm.head(bb.output_norm(h))


def make_body(ctx, model, x, y):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, capturable=ctx.cuda)

    def body():
        with ctx.amp(cache=False):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    return body, opt


def graph_step(ctx, model, x, y):
    body, opt = make_body(ctx, model, x, y)
    stream = torch.cuda.Stream(ctx.device)
    stream.wait_stream(torch.cuda.current_stream(ctx.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            body()
    torch.cuda.current_stream(ctx.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        body()
    return graph.replay


def dynamo_counters():
    from torch._dynamo.utils import counters
    return counters


def run_variant(ctx, variant, base, x, y):
    model = copy.deepcopy(base)
    rec = {"t": [round(time.time(), 2)]}
    if ctx.cuda:
        torch.cuda.reset_peak_memory_stats(ctx.device)
    if variant in ("eager", "eager_late", "eager_cpu_busy"):
        opt, scaler = trainer(ctx, model)
        rec["step"] = timed(ctx, lambda: train_step(ctx, model, opt, scaler, x, y))
    elif variant == "eager_body":
        body, opt = make_body(ctx, model, x, y)
        rec["step"] = timed(ctx, lambda: (opt.zero_grad(set_to_none=True), body()))
    elif variant == "eager_fwdbwd":
        def fb():
            model.zero_grad(set_to_none=True)
            with ctx.amp():
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            loss.backward()
        rec["step"] = timed(ctx, fb)
    elif variant in ("ptr_outside", "ptr_outside_cpu_busy"):
        bb = model.backbone
        wrapped = PtrOutside(model, match_pointers(x, bb.ptr, bb.vocab))
        opt, scaler = trainer(ctx, wrapped)
        rec["step"] = timed(ctx, lambda: train_step(ctx, wrapped, opt, scaler, x, y))
    elif variant == "graph_step":
        t0 = time.perf_counter()
        replay = graph_step(ctx, model, x, y)
        rec["capture_s"] = round(time.perf_counter() - t0, 2)
        rec["step"] = timed(ctx, replay)
    elif variant in ("compile", "compile_ro"):
        torch._dynamo.reset()
        counters = dynamo_counters()
        counters.clear()
        rec["cache"] = "warm" if ctx.compiled_once else "cold"
        ctx.compiled_once = True
        compiled = torch.compile(model) if variant == "compile" else torch.compile(model, mode="reduce-overhead")
        opt, scaler = trainer(ctx, model)
        mark = torch.compiler.cudagraph_mark_step_begin if variant == "compile_ro" else (lambda: None)
        t0 = time.perf_counter()
        mark()
        train_step(ctx, compiled, opt, scaler, x, y)
        ctx.sync()
        rec["first_step_s"] = round(time.perf_counter() - t0, 2)
        rec["step"] = timed(ctx, lambda: (mark(), train_step(ctx, compiled, opt, scaler, x, y)), warmup=3)
        rec["counters"] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in counters.items() if v}
        rec["cudagraph_skips"] = int(counters["inductor"].get("cudagraph_skips", 0))
        rec["graph_breaks"] = int(sum(counters["graph_break"].values()))
    if ctx.cuda:
        rec["peak_mb"] = round(torch.cuda.max_memory_allocated(ctx.device) / 2 ** 20, 1)
    rec["t"].append(round(time.time(), 2))
    return rec


def cpu_busy():
    code = ("import torch\nfrom wat.common import match_pointers\ntorch.set_num_threads(1)\n"
            "x = torch.randint(0, 256, (32, 512))\nwhile True:\n    match_pointers(x, [4, 8, 16, 32], 256)\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    return proc


def test_d(ctx):
    plan = [("main", 512), ("tfm", 512), ("combo", 512), ("main", 2048)] if not ctx.quick else [("main", ctx.lengths[0])]
    records, bases = [], {}
    for kind, T in plan:
        rec = {"kind": kind, "size": "1m", "T": T, "variants": {}}
        records.append(rec)
        try:
            base, width = make_model(ctx, kind, "1m", T)
            x, y = ctx.batch(T)
            rec.update(B=x.size(0), D=width)
            bases[(kind, T)] = (base, x, y)
        except Exception:
            rec["error"] = error_text()

    def run(rec, variant):
        key = (rec["kind"], rec["T"])
        if key not in bases:
            return
        if variant.startswith("ptr_outside") and rec["kind"] != "main":
            return
        if not ctx.cuda and variant in ("graph_step", "compile", "compile_ro"):
            return
        if variant == "compile_ro" and "error" in rec["variants"].get("compile", {}):
            rec["variants"][variant] = {"skipped": "compile failed"}
        elif ctx.left() < (480 if variant.startswith("compile") else 60):
            rec["variants"][variant] = {"skipped": "time"}
        else:
            base, x, y = bases[key]
            try:
                rec["variants"][variant] = run_variant(ctx, variant, base, x, y)
            except Exception:
                rec["variants"][variant] = {"error": error_text()}
            if variant.startswith("compile"):
                torch._dynamo.reset()
            ctx.free()
        save(ctx, "D", {"semantics": SEMANTICS, "records": records})
        got = rec["variants"][variant]
        extra = f" skips {got.get('cudagraph_skips')} breaks {got.get('graph_breaks')}" if variant.startswith("compile") else ""
        print(f"[D] {rec['kind']} T={rec['T']} {variant}: {got.get('step', {}).get('median_ms')} ms{extra}"
              f"{' ' + got['skipped'] if 'skipped' in got else ''}{' ERROR ' + got['error'][-300:] if 'error' in got else ''}", flush=True)

    for rec in records:
        for variant in ("eager", "eager_body", "eager_fwdbwd", "ptr_outside", "graph_step"):
            run(rec, variant)
    if records and records[0]["kind"] == "main" and not ctx.quick:
        busy = cpu_busy()
        try:
            for variant in ("eager_cpu_busy", "ptr_outside_cpu_busy"):
                run(records[0], variant)
        finally:
            busy.kill()
            busy.wait()
    if ctx.cuda:
        try:
            torch._logging.set_logs(perf_hints=True, graph_breaks=True)
        except Exception:
            pass
        for rec in records[:3]:
            for variant in ("compile", "compile_ro"):
                run(rec, variant)
    for rec in records:
        run(rec, "eager_late")
    bases.clear()
    ctx.free()
    return records


def test_e(ctx):
    records = []
    threads = torch.get_num_threads()
    for T in ctx.lengths:
        B = max(1, ctx.tokens // T)
        x = torch.randint(0, VOCAB, (B, T), generator=torch.Generator().manual_seed(1))
        rec = {"T": T, "B": B, "t": [round(time.time(), 2)]}
        try:
            xd = x.to(ctx.device)
            rec["device_ms"] = throughput(ctx, lambda: match_pointers(xd, list(PTR), VOCAB))
            ref = [c.cpu() for c in match_pointers(xd, list(PTR), VOCAB)]
            got = ref
            for n in sorted({1, 2, min(4, os.cpu_count() or 1)}):
                torch.set_num_threads(n)
                reps = 2 if ctx.quick else 5
                t0 = time.perf_counter()
                for _ in range(reps):
                    got = match_pointers(x, list(PTR), VOCAB)
                rec[f"cpu_ms_threads{n}"] = round((time.perf_counter() - t0) * 1e3 / reps, 3)
            rec["cpu_equals_device"] = all(torch.equal(a, b) for a, b in zip(ref, got))
            rec["hit_rate"] = [round(float((c < VOCAB).float().mean()), 4) for c in got]
        except Exception:
            rec["error"] = error_text()
        finally:
            torch.set_num_threads(threads)
        rec["t"].append(round(time.time(), 2))
        records.append(rec)
        save(ctx, "E", {"semantics": SEMANTICS, "records": records})
        print(f"[E] T={T}: device {rec.get('device_ms')} ms, cpu(1) {rec.get('cpu_ms_threads1')} ms", flush=True)
    return records


def calibrate(ctx):
    if not ctx.cuda:
        return {}
    out = {}
    try:
        a = torch.empty(2 ** 28, dtype=torch.uint8, device=ctx.device)
        b = torch.empty_like(a)
        out["copy_gbps"] = round(2 * 2 ** 28 / (throughput(ctx, lambda: b.copy_(a), warmup=3, iters=10) * 1e-3) / 1e9, 1)
        del a, b
        m = torch.randn(4096, 4096, device=ctx.device, dtype=torch.float16)
        out["fp16_tflops"] = round(2 * 4096 ** 3 / (throughput(ctx, lambda: m @ m, warmup=3, iters=10) * 1e-3) / 1e12, 2)
        del m
    except Exception:
        out["error"] = error_text()
    ctx.free()
    return out


TESTS = {"A": test_a, "B": test_b, "C": test_c, "D": test_d, "E": test_e}


def gpu_clocks():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile WAT 1.0 against the Transformer, part by part.")
    parser.add_argument("--tests", default="A,B,C,E,D")
    parser.add_argument("--out", default=os.path.join("results", "speed"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--minutes", type=float, default=50.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    tests = [t.strip().upper() for t in args.tests.split(",") if t.strip()]
    if "D" in tests and any(t in tests[tests.index("D") + 1:] for t in ("A", "B", "C")):
        raise SystemExit("test D uses CUDA graphs and must run after the profiled tests A, B and C")
    os.makedirs(args.out, exist_ok=True)
    faulthandler.dump_traceback_later(args.minutes * 60 + 300, exit=True)
    ctx = Ctx(args)
    clocks = Clocks() if ctx.cuda else None
    torch.manual_seed(0)
    info = {"env": environment(ctx.device), "precision": ctx.precision, "tokens_per_step": ctx.tokens,
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "clocks_start": gpu_clocks(),
            "semantics": SEMANTICS, "tests": {}}
    info["calibration"] = ctx.calib = calibrate(ctx)
    save(ctx, "env", info)
    failed = False
    try:
        for name in tests:
            t0 = time.time()
            print(f"=== test {name}", flush=True)
            try:
                skipped = count_skipped(TESTS[name](ctx))
                status = "done" + (f" ({skipped} skipped)" if skipped else "")
            except Exception:
                status = error_text()
                failed = True
                print(f"[{name}] FAILED\n{status}", flush=True)
            info["tests"][name] = {"status": status, "seconds": round(time.time() - t0, 1)}
            info["clocks_end"] = gpu_clocks()
            save(ctx, "env", info)
            if clocks is not None:
                save(ctx, "clocks", clocks.snapshot())
    finally:
        if clocks is not None:
            save(ctx, "clocks", clocks.stop())
        faulthandler.cancel_dump_traceback_later()
    print("=== speed profile finished", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
