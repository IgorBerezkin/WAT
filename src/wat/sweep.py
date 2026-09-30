import argparse
import itertools
import json
import os
import sys
import time
import traceback

from wat.run import merge, resolve, run, run_name, set_path


def expand(spec):
    base = spec.get("base", {})
    grid = spec.get("grid", {})
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    configs = []
    for override in spec.get("runs", [{}]):
        for combo in combos:
            cfg = merge(base, override)
            for key, value in zip(keys, combo):
                set_path(cfg, key, value)
            configs.append(cfg)
    names = [run_name(resolve(c)) for c in configs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate runs in sweep: {sorted(duplicates)}")
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a sweep of WAT experiments.")
    parser.add_argument("spec")
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=None, metavar="HOURS")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    with open(args.spec, encoding="utf-8") as f:
        configs = expand(json.load(f))[args.shard::args.num_shards]
    deadline = time.time() + args.time_limit * 3600 if args.time_limit else None
    if args.dry_run:
        for cfg in configs:
            print(run_name(resolve(cfg)))
        return 0
    status = {"done": 0, "interrupted": 0, "failed": 0}
    for i, cfg in enumerate(configs, 1):
        if deadline is not None and time.time() > deadline:
            status["interrupted"] += len(configs) - i + 1
            break
        print(f"=== run {i}/{len(configs)} (shard {args.shard}/{args.num_shards})", flush=True)
        try:
            result = run(cfg, args.out, device=args.device, deadline=deadline,
                         log=lambda msg: print(msg, flush=True))
            status[result.get("status", "done")] += 1
        except Exception:
            status["failed"] += 1
            name = run_name(resolve(cfg))
            os.makedirs(os.path.join(args.out, name), exist_ok=True)
            with open(os.path.join(args.out, name, "error.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            traceback.print_exc()
    print(f"sweep finished: {status}", flush=True)
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
