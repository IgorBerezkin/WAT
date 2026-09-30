import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from wat.sweep import expand

REPO = "IgorBerezkin/WAT"
MACHINES = {"t4x2": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100", "cpu": None}
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}

KERNEL_SCRIPT = '''import json
import os
import subprocess
import sys
import tarfile
import urllib.request

import torch

REPO = @REPO@
COMMIT = @COMMIT@
SPEC = @SPEC@
HOURS = @HOURS@

output_dir = os.environ.get("WAT_OUTPUT_DIR", "/kaggle/working")
work_dir = os.environ.get("WAT_WORK_DIR", "/kaggle/temp")
os.makedirs(work_dir, exist_ok=True)
archive = os.path.join(work_dir, "code.tar.gz")
urllib.request.urlretrieve(f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz", archive)
with tarfile.open(archive) as tar:
    try:
        tar.extractall(work_dir, filter="data")
    except TypeError:
        tar.extractall(work_dir)
code_dir = os.path.join(work_dir, "WAT-" + COMMIT)
spec_path = os.path.join(work_dir, "spec.json")
with open(spec_path, "w") as f:
    json.dump(SPEC, f)
env = dict(os.environ, PYTHONPATH=os.path.join(code_dir, "src"), WAT_COMMIT=COMMIT,
           WAT_DATA_DIR=os.path.join(work_dir, "data"))
subprocess.run([sys.executable, "-c", "from wat.data import read_shakespeare; read_shakespeare()"],
               env=env, check=True)
gpus = torch.cuda.device_count()
shards = max(1, gpus)
runs_dir = os.path.join(output_dir, "runs")
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
procs = []
for i in range(shards):
    shard_env = dict(env, CUDA_VISIBLE_DEVICES=str(i)) if gpus else env
    log = open(os.path.join(logs_dir, f"shard{i}.log"), "w")
    cmd = [sys.executable, "-m", "wat.sweep", spec_path, "--out", runs_dir,
           "--shard", str(i), "--num-shards", str(shards), "--time-limit", str(HOURS)]
    procs.append((subprocess.Popen(cmd, env=shard_env, stdout=log, stderr=subprocess.STDOUT), log))
codes = []
for proc, log in procs:
    codes.append(proc.wait())
    log.close()
subprocess.run([sys.executable, "-m", "wat.report", runs_dir, "--out",
                os.path.join(output_dir, "summary.md")], env=env)
with open(os.path.join(output_dir, "job.json"), "w") as f:
    json.dump({"commit": COMMIT, "shards": shards, "exit_codes": codes,
               "gpus": [torch.cuda.get_device_name(i) for i in range(gpus)]}, f, indent=1)
for i in range(shards):
    with open(os.path.join(logs_dir, f"shard{i}.log")) as f:
        print(f.read()[-4000:])
'''


def kaggle(*args):
    local = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
    exe = local if os.path.exists(local) else shutil.which("kaggle")
    if exe is None:
        raise SystemExit("kaggle CLI not found: pip install kaggle")
    out = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0:
        raise SystemExit(text)
    return text


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def username(explicit=None):
    user = explicit or os.environ.get("WAT_KAGGLE_USER")
    if user:
        return user
    match = re.search(r"^([\w-]+)/[\w-]+\s", kaggle("kernels", "list", "--mine", "--page-size", "1"), re.M)
    if not match:
        raise SystemExit("cannot detect Kaggle username; pass --user")
    return match.group(1)


def resolve_commit(ref):
    sha = git("rev-parse", ref)
    if not git("branch", "-r", "--contains", sha):
        raise SystemExit(f"commit {sha[:7]} is not on GitHub; push it first")
    if git("status", "--porcelain", "--untracked-files=no", "--", "src"):
        print("warning: uncommitted changes in src/ are not part of the job", flush=True)
    return sha


def build(spec, sha, user, slug, machine, hours, root):
    kernel_dir = os.path.join(root, slug)
    os.makedirs(kernel_dir, exist_ok=True)
    script = (KERNEL_SCRIPT.replace("@REPO@", repr(REPO)).replace("@COMMIT@", repr(sha))
              .replace("@SPEC@", repr(spec)).replace("@HOURS@", repr(hours)))
    with open(os.path.join(kernel_dir, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    metadata = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": machine is not None,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [],
                "kernel_sources": []}
    if machine:
        metadata["machine_shape"] = machine
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1)
    return kernel_dir


def status(user, slug):
    text = kaggle("kernels", "status", f"{user}/{slug}")
    match = re.search(r'status "([^"]+)"', text)
    return (match.group(1) if match else text).split(".")[-1].lower()


def fetch(user, slug, out):
    os.makedirs(out, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{slug}", "-p", out)
    summary = os.path.join(out, "summary.md")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            print(f.read())
    print(f"outputs: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run WAT sweeps on Kaggle GPUs.")
    parser.add_argument("--user", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit")
    p.add_argument("spec")
    p.add_argument("--slug", required=True)
    p.add_argument("--commit", default="HEAD")
    p.add_argument("--machine", default="t4x2", choices=sorted(MACHINES))
    p.add_argument("--hours", type=float, default=11.0)
    p.add_argument("--build-dir", default=os.path.join("build", "kaggle"))
    p.add_argument("--dry-run", action="store_true")
    for name in ("status", "fetch", "wait"):
        p = sub.add_parser(name)
        p.add_argument("slug")
        p.add_argument("--out", default=None)
        p.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.command == "submit":
        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)
        n_runs = len(expand(spec))
        sha = resolve_commit(args.commit)
        user = username(args.user)
        kernel_dir = build(spec, sha, user, args.slug, MACHINES[args.machine], args.hours,
                           args.build_dir)
        print(f"{user}/{args.slug}: {n_runs} runs, commit {sha[:7]}, machine {args.machine}, "
              f"files in {kernel_dir}", flush=True)
        if not args.dry_run:
            print(kaggle("kernels", "push", "-p", kernel_dir))
        return 0

    user = username(args.user)
    out = args.out or os.path.join("results", "remote", args.slug)
    if args.command == "status":
        print(status(user, args.slug))
        return 0
    if args.command == "fetch":
        fetch(user, args.slug, out)
        return 0
    last = None
    while True:
        state = status(user, args.slug)
        if state != last:
            print(f"{time.strftime('%H:%M:%S')} {state}", flush=True)
            last = state
        if state in TERMINAL:
            break
        time.sleep(args.interval)
    fetch(user, args.slug, out)
    return 0 if state == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
