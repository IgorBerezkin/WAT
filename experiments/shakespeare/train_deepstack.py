import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from wat.models.wat_deepstack import WATDeepStackV1, count_params, find_embed_dim, generate_text
from wat.models.transformer_baseline import TransformerBaseline, find_embed_dim_transformer

import warnings
warnings.filterwarnings("ignore", message="Mismatch dtype")

np.random.seed(42)
torch.manual_seed(42)


def load_shakespeare(path="/tmp/shakespeare.txt"):
    try:
        with open(path) as f:
            text = f.read()
    except:
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, path)
        with open(path) as f:
            text = f.read()
    return text, len(text)


def build_vocab(text):
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    return vocab, len(chars)


class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def train(model, train_loader, test_loader, epochs, lr, weight_decay, device, vocab, idx_to_char, n_layers, chunk_size, dropout):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0

    history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        num_batches = 0
        train_tokens = 0
        train_loss_sum = 0
        train_tokens = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.numel()
            train_tokens += y.numel()

        train_loss = train_loss_sum / train_tokens
        scheduler.step()

        model.eval()

        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(train_loader):
                if i >= 50:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(-1)
                train_correct += (pred == y).sum().item()
                train_total += y.numel()
        train_acc = train_correct / train_total if train_total > 0 else 0

        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction='sum'
                )
                val_loss += loss.item()

                pred = logits.argmax(-1)
                val_correct += (pred == y).sum().item()
                val_total += y.numel()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        overfit = train_acc - val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1

        epoch_time = time.time() - epoch_start

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'overfit': overfit,
            'time': epoch_time
        })

        print(f"  {epoch+1}/{epochs}  |  {train_loss:.4f}  |  {val_loss:.4f}  |  {train_acc*100:5.2f}%  |  {val_acc*100:5.2f}%  |  {overfit*100:+5.2f}%  |  {epoch_time:.1f}s")

        if epoch == epochs - 1:
            print(f"\n  === Generated text after epoch {epoch+1} ===")
            gen = generate_text(model, [vocab.get(c, 0) for c in "First"], idx_to_char, device, max_len=200)
            print(f"  {gen[:300]}...")
            print(f"  === End ===\n")

    return best_val_acc, best_val_loss, best_epoch, history


def main():
    print("=" * 80)
    print("CONFIG")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text, n_chars = load_shakespeare()
    vocab, vocab_size = build_vocab(text)
    idx_to_char = {i: c for c, i in vocab.items()}

    train_size = 50_000
    seq_len = 512
    batch_size = 32
    epochs = 10
    lr = 0.0003
    weight_decay = 0.01
    target_params = 50_000
    n_layers = 1
    chunk_size = 32
    dropout = 0.2

    data = np.array([vocab.get(c, 0) for c in text], dtype=np.int64)

    train_end = train_size
    test_start = train_end + seq_len
    test_end = test_start + 5_000

    print(f"  Dataset:       Shakespeare ({n_chars:,} chars)")
    print(f"  Vocab size:    {vocab_size}")
    print(f"  Train:         0 - {train_end:,} ({train_end:,} samples)")
    print(f"  Val:           {test_start:,} - {test_end:,} ({test_end - test_start:,} samples)")
    print(f"  seq_len:       {seq_len}")
    print(f"  batch_size:    {batch_size}")
    print(f"  epochs:        {epochs}")
    print(f"  lr:            {lr}")
    print(f"  weight_decay:  {weight_decay}")

    train_ds = ShakespeareDataset(data[:train_end], seq_len)
    test_ds = ShakespeareDataset(data[test_start:test_end], seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print("\n" + "=" * 80)
    print("WAT MODEL")
    print("=" * 80)

    ed, n_params = find_embed_dim(vocab_size, target_params, n_layers=n_layers)

    print(f"  Architecture:  WATDeepStackV1")
    print(f"  embed_dim:    {ed}")
    print(f"  n_layers:     {n_layers}")
    print(f"  chunk_size:   {chunk_size}")
    print(f"  dropout:      {dropout}")
    print(f"  params:       {n_params:,}")

    model = WATDeepStackV1(
        vocab_size, ed,
        n_layers=n_layers,
        chunk_size=chunk_size,
        dropout=dropout
    )
    actual_params = count_params(model)

    print("\n" + "=" * 80)
    print("TRAINING WAT")
    print("=" * 80)
    print(f"  Epoch | Train Loss | Val Loss  | Train Acc | Val Acc   | Overfit | Time")
    print(f"  ------|------------|-----------|-----------|-----------|---------|------")

    best_val_acc, best_val_loss, best_epoch, history = train(
        model, train_loader, test_loader, epochs, lr, weight_decay,
        device, vocab, idx_to_char, n_layers, chunk_size, dropout
    )

    print("\n" + "=" * 80)
    print("TRANSFORMER MODEL")
    print("=" * 80)

    t_ed, t_params = find_embed_dim_transformer(vocab_size, target_params, n_layers=n_layers)
    trans_model = TransformerBaseline(
        vocab_size, t_ed,
        n_layers=n_layers,
        chunk_size=chunk_size,
        dropout=dropout
    )
    t_actual_params = trans_model.count_params()

    print(f"  Architecture:  TransformerBaseline")
    print(f"  embed_dim:    {t_ed}")
    print(f"  n_layers:     {n_layers}")
    print(f"  dropout:      {dropout}")
    print(f"  params:       {t_actual_params:,}")

    print("\n" + "=" * 80)
    print("TRAINING TRANSFORMER")
    print("=" * 80)
    print(f"  Epoch | Train Loss | Val Loss  | Train Acc | Val Acc   | Overfit | Time")
    print(f"  ------|------------|-----------|-----------|-----------|---------|------")

    t_best_val_acc, t_best_val_loss, t_best_epoch, t_history = train(
        trans_model, train_loader, test_loader, epochs, lr, weight_decay,
        device, vocab, idx_to_char, n_layers, chunk_size, dropout
    )

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ: WAT vs TRANSFORMER")
    print("=" * 80)
    print(f"  {'Модель':<22} {'Params':>8} {'Val Acc':>10} {'Val Loss':>10} {'Best Epoch':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'WATDeepStackV1':<22} {actual_params:>8,} {best_val_acc*100:>9.2f}% {best_val_loss:>10.4f} {best_epoch:>12}")
    print(f"  {'TransformerBaseline':<22} {t_actual_params:>8,} {t_best_val_acc*100:>9.2f}% {t_best_val_loss:>10.4f} {t_best_epoch:>12}")
    winner = "WAT" if best_val_acc > t_best_val_acc else "Transformer"
    diff = abs(best_val_acc - t_best_val_acc) * 100
    print(f"\n  Победитель: {winner}  (+{diff:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
