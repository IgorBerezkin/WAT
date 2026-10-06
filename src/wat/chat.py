import argparse
import codecs
import glob
import json
import os
import sys
import time

import torch

from wat.data import BOT, END, chat_bytes
from wat.run import atomic_json, build_model

PROMPTS = (
    "Привет! Расскажи, кто ты.",
    "Как приготовить омлет?",
    "Напиши короткое стихотворение про осень.",
    "Объясни простыми словами, что такое нейросеть.",
    "Дай три совета, как лучше учиться.",
    "Почему небо голубое?",
    "Переведи на английский: «Сегодня хорошая погода».",
    "Составь список покупок для пикника.",
    "Чем кошка отличается от собаки?",
    "Придумай название для кафе у моря.",
)


def find_run(pattern, bases):
    candidates = [pattern] if os.path.isabs(pattern) else [os.path.join(b, pattern) for b in bases]
    for candidate in candidates:
        found = sorted(p for p in glob.glob(candidate) if os.path.exists(os.path.join(p, "model.pt")))
        if found:
            return found[-1]
    raise SystemExit(f"no trained model matches {pattern}")


def load(run_dir, device):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    mcfg = dict(cfg["model"])
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            mcfg["embed_dim"] = json.load(f)["embed_dim"]
    model, _ = build_model(mcfg, 256, cfg["task"]["seq_len"], cfg["seed"])
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg["task"]["seq_len"]


def amp(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def generate(model, context, seq_len, device, max_new=600, temperature=0.8, top_k=40, on_byte=None):
    data = list(context)
    out = []
    for _ in range(max_new):
        x = torch.tensor(data[-seq_len:], dtype=torch.long, device=device)[None]
        with amp(device):
            logits = model(x)[0, -1].float()
        logits = logits / max(temperature, 1e-4)
        if top_k:
            values, index = logits.topk(top_k)
            choice = index[torch.multinomial(torch.softmax(values, -1), 1)]
        else:
            choice = torch.multinomial(torch.softmax(logits, -1), 1)
        b = int(choice)
        if b == END:
            break
        data.append(b)
        out.append(b)
        if on_byte:
            on_byte(b)
    return bytes(out)


def samples(model, seq_len, device, out_dir, args):
    records = []
    for prompt in PROMPTS[:args.n]:
        t0 = time.perf_counter()
        torch.manual_seed(args.seed)
        answer = generate(model, chat_bytes(prompt), seq_len, device, args.max_new, args.temperature, args.top_k)
        records.append({"prompt": prompt, "answer": answer.decode("utf-8", errors="replace"),
                        "bytes": len(answer), "seconds": round(time.perf_counter() - t0, 2)})
        print(f"Q: {prompt}\nA: {records[-1]['answer']}\n", flush=True)
        atomic_json(os.path.join(out_dir, "chat_samples.json"),
                    {"temperature": args.temperature, "top_k": args.top_k, "samples": records})
    return records


def talk(model, seq_len, device, args):
    history = b""
    print("WAT 1.0 чат. Пустая строка — выход, /new — начать заново.", flush=True)
    while True:
        try:
            question = input("\nВы: ").strip()
        except EOFError:
            break
        if not question:
            break
        if question == "/new":
            history = b""
            continue
        context = history + chat_bytes(question)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        sys.stdout.write("WAT: ")
        sys.stdout.flush()

        def show(b):
            sys.stdout.write(decoder.decode(bytes([b])))
            sys.stdout.flush()

        answer = generate(model, context, seq_len, device, args.max_new, args.temperature, args.top_k, show)
        sys.stdout.write(decoder.decode(b"", final=True) + "\n")
        history = (context + answer + bytes([END]) + b"\n")[-seq_len:]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Talk to a trained WAT model.")
    parser.add_argument("run", nargs="?", default="runs/chat-*")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--n", type=int, default=len(PROMPTS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bases = [os.getcwd()]
    if args.out:
        out = os.path.abspath(args.out)
        bases = [os.path.dirname(out), os.path.dirname(os.path.dirname(out))] + bases
    run_dir = find_run(args.run, bases)
    model, seq_len = load(run_dir, device)
    print(f"model: {run_dir} on {device}", flush=True)
    if args.samples:
        out_dir = args.out or run_dir
        os.makedirs(out_dir, exist_ok=True)
        samples(model, seq_len, device, out_dir, args)
    else:
        talk(model, seq_len, device, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
