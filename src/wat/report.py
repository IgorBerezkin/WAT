import argparse
import json
import os
import statistics
import sys


def collect(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        if "metrics.json" in files:
            with open(os.path.join(dirpath, "metrics.json"), encoding="utf-8") as f:
                metrics = json.load(f)
            if metrics.get("status") == "done":
                rows.append(metrics)
    return rows


def describe_task(task):
    extras = [f"{k}={v}" for k, v in sorted(task.items()) if k != "name"]
    return task["name"] + (f" ({', '.join(extras)})" if extras else "")


def describe_model(metrics):
    model = metrics["config"]["model"]
    name = model["name"]
    if name == "wat":
        name += f"[{model['ctx_mode']}{'+intra' if model.get('intra') else ''}]"
    if name == "ngram":
        return f"ngram[n={model['order']}]"
    return f"{name} d={metrics.get('embed_dim')} L={model.get('n_layers')}"


def mean_std(values):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(rows):
    groups = {}
    for metrics in rows:
        groups.setdefault((describe_task(metrics["config"]["task"]), metrics["group"]), []).append(metrics)
    table = []
    for (task, group), items in groups.items():
        entry = {"task": task, "group": group, "model": describe_model(items[0]),
                 "params": items[0].get("params"), "seeds": len(items),
                 "train_time_s": statistics.fmean(m.get("train_time_s", 0.0) for m in items)}
        for key in ("val_bpc", "test_bpc", "test_acc"):
            entry[key] = mean_std([m["result"][key] for m in items])
        table.append(entry)
    table.sort(key=lambda e: (e["task"], e["test_bpc"][0]))
    return table


def fmt(pair, scale=1.0, digits=4):
    mean, std = pair
    if mean is None:
        return "—"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def markdown(table):
    lines = []
    for task in dict.fromkeys(e["task"] for e in table):
        lines += [f"### {task}", "",
                  "| model | params | seeds | test bpc | test acc, % | val bpc | train time, s |",
                  "|---|---|---|---|---|---|---|"]
        for e in (e for e in table if e["task"] == task):
            params = f"{e['params']:,}" if e["params"] else "—"
            lines.append(f"| {e['model']} | {params} | {e['seeds']} | {fmt(e['test_bpc'])} | "
                         f"{fmt(e['test_acc'], 100, 2)} | {fmt(e['val_bpc'])} | "
                         f"{e['train_time_s']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate WAT experiment results.")
    parser.add_argument("root", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    rows = [m for root in args.root for m in collect(root)]
    text = markdown(aggregate(rows)) if rows else "no finished runs\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
