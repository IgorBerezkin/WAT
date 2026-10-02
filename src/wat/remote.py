import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

from wat.run import resolve
from wat.sweep import expand

REPO = "IgorBerezkin/WAT"
MACHINES = {"t4x2": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100", "cpu": None}
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}

KERNEL_SCRIPT = '''import base64
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

import torch

REPO = @REPO@
COMMIT = @COMMIT@
CODE = @CODE@
SPEC = @SPEC@
HOURS = @HOURS@
PREFETCH = @PREFETCH@

output_dir = os.environ.get("WAT_OUTPUT_DIR", "/kaggle/working")
work_dir = os.environ.get("WAT_WORK_DIR", "/kaggle/temp")
os.makedirs(work_dir, exist_ok=True)
archive = os.path.join(work_dir, "code.tar.gz")
if CODE:
    code_dir = os.path.join(work_dir, "code")
    with open(archive, "wb") as f:
        f.write(base64.b64decode(CODE))
else:
    code_dir = os.path.join(work_dir, "WAT-" + COMMIT)
    urllib.request.urlretrieve(f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz", archive)
with tarfile.open(archive) as tar:
    target = code_dir if CODE else work_dir
    try:
        tar.extractall(target, filter="data")
    except TypeError:
        tar.extractall(target)
env = dict(os.environ, PYTHONPATH=os.path.join(code_dir, "src"), WAT_COMMIT=COMMIT,
           WAT_DATA_DIR=os.path.join(work_dir, "data"), PYTHONUNBUFFERED="1")
for dataset in PREFETCH:
    subprocess.run([sys.executable, "-c", f"from wat.data import read_{dataset}; read_{dataset}()"],
                   env=env, check=True)
gpus = torch.cuda.device_count()
workers = max(1, gpus)
started = time.time()
token = f"{COMMIT[:12]}-{int(started)}"
runs_dir = os.path.join(output_dir, "runs")
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)


def pump(proc, log, tag):
    for line in proc.stdout:
        log.write(line)
        log.flush()
        print(f"[{tag}] {line}", end="", flush=True)


def run_stage(spec_path, index):
    remaining = max(0.05, HOURS - (time.time() - started) / 3600)
    procs = []
    for i in range(workers):
        worker_env = dict(env, CUDA_VISIBLE_DEVICES=str(i)) if gpus else env
        log = open(os.path.join(logs_dir, f"stage{index}-gpu{i}.log"), "w")
        cmd = [sys.executable, "-m", "wat.sweep", spec_path, "--out", runs_dir,
               "--claim", f"{token}-s{index}", "--time-limit", str(remaining)]
        proc = subprocess.Popen(cmd, env=worker_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        thread = threading.Thread(target=pump, args=(proc, log, f"gpu{i}"), daemon=True)
        thread.start()
        procs.append((proc, log, thread))
    codes = []
    for proc, log, thread in procs:
        codes.append(proc.wait())
        thread.join()
        log.close()
    return codes


stages = SPEC if isinstance(SPEC, list) else [SPEC]
stage_codes = []
for index, stage in enumerate(stages):
    spec_path = os.path.join(output_dir, f"stage{index}.json")
    if "generator" in stage:
        module, func = stage["generator"].split(":")
        generate = (f"import importlib, json; spec = getattr(importlib.import_module({module!r}), "
                    f"{func!r})({runs_dir!r}, **{stage.get('params', {})!r}); "
                    f"json.dump(spec, open({spec_path!r}, 'w'), indent=1)")
        subprocess.run([sys.executable, "-c", generate], env=env, check=True)
    else:
        with open(spec_path, "w") as f:
            json.dump(stage, f, indent=1)
    print(f"=== stage {index + 1}/{len(stages)} started", flush=True)
    stage_codes.append(run_stage(spec_path, index))
subprocess.run([sys.executable, "-m", "wat.report", runs_dir, "--out",
                os.path.join(output_dir, "summary.md")], env=env)
with open(os.path.join(output_dir, "job.json"), "w") as f:
    json.dump({"commit": COMMIT, "workers": workers, "exit_codes": stage_codes,
               "hours": round((time.time() - started) / 3600, 2),
               "gpus": [torch.cuda.get_device_name(i) for i in range(gpus)]}, f, indent=1)
print("=== JOB FINISHED ===", flush=True)
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


def pack_code():
    root = git("rev-parse", "--show-toplevel")
    files = []
    for dirpath, dirnames, names in os.walk(os.path.join(root, "src", "wat")):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        files += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".py")]
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files:
            arcname = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "rb") as f:
                digest.update(arcname.encode() + b"\0" + f.read() + b"\0")
            tar.add(path, arcname=arcname)
    return buffer.getvalue(), digest.hexdigest()


def stages_of(spec):
    return spec if isinstance(spec, list) else [spec]


def prefetch(spec):
    names = set()
    for stage in stages_of(spec):
        if "generator" in stage:
            names.update(stage.get("prefetch", []))
        else:
            names.update(resolve(cfg)["task"]["name"] for cfg in expand(stage))
    return sorted(names & {"shakespeare", "enwik8"})


def build(spec, label, code, user, slug, machine, hours, root):
    kernel_dir = os.path.join(root, slug)
    os.makedirs(kernel_dir, exist_ok=True)
    code_b64 = base64.b64encode(code).decode("ascii") if code else None
    script = (KERNEL_SCRIPT.replace("@REPO@", repr(REPO)).replace("@COMMIT@", repr(label))
              .replace("@CODE@", repr(code_b64)).replace("@SPEC@", repr(spec))
              .replace("@HOURS@", repr(hours)).replace("@PREFETCH@", repr(prefetch(spec))))
    with open(os.path.join(kernel_dir, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    if code:
        with open(os.path.join(kernel_dir, "code.tar.gz"), "wb") as f:
            f.write(code)
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
    return (match.group(1) if match else text).split(".")[-1].lower().replace("_", "")


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
    p.add_argument("--code", default="local", choices=["local", "commit"])
    p.add_argument("--machine", default="t4x2", choices=sorted(MACHINES))
    p.add_argument("--hours", type=float, default=11.0)
    p.add_argument("--build-dir", default=os.path.join("build", "kaggle"))
    p.add_argument("--dry-run", action="store_true")
    for name in ("status", "fetch", "wait", "logs"):
        p = sub.add_parser(name)
        p.add_argument("slug")
        p.add_argument("--out", default=None)
        p.add_argument("--interval", type=float, default=60.0)
        p.add_argument("--tail", type=int, default=20)
    args = parser.parse_args(argv)

    if args.command == "submit":
        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)
        stages = stages_of(spec)
        n_runs = sum(len(expand(s)) for s in stages if "generator" not in s)
        generated = sum("generator" in s for s in stages)
        if args.code == "commit":
            label, code = resolve_commit(args.commit), None
        else:
            code, digest = pack_code()
            label = f"{git('rev-parse', 'HEAD')}+local-{digest[:12]}"
        user = username(args.user)
        kernel_dir = build(spec, label, code, user, args.slug, MACHINES[args.machine],
                           args.hours, args.build_dir)
        extra = f" + {generated} generated stage(s)" if generated else ""
        print(f"{user}/{args.slug}: {n_runs} runs{extra}, code {label[:7]}"
              f"{label[40:] if code else ''}, machine {args.machine}, files in {kernel_dir}",
              flush=True)
        if not args.dry_run:
            print(kaggle("kernels", "push", "-p", kernel_dir))
        return 0

    user = username(args.user)
    out = args.out or os.path.join("results", "remote", args.slug)
    if args.command == "status":
        print(status(user, args.slug))
        return 0
    if args.command == "logs":
        exe = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
        try:
            out = subprocess.run([exe, "kernels", "logs", "-f", f"{user}/{args.slug}"],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=args.interval).stdout
        except subprocess.TimeoutExpired as stopped:
            out = stopped.stdout or ""
            out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        print("\n".join(out.splitlines()[-args.tail:]))
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
