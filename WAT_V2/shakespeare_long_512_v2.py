import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from WAT_v2 import WATModel

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

class ShakespeareSeq2SeqDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads=4, max_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.predict = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask)
        return self.predict(x)

def find_embed_dim(model_class, vocab_size, target_params):
    best_ed, best_params = None, float('inf')
    for ed in range(8, 256, 4):
        model = model_class(vocab_size, ed)
        n_params = sum(p.numel() for p in model.parameters())
        if abs(n_params - target_params) < abs(best_params - target_params):
            best_params = n_params
            best_ed = ed
    return best_ed, best_params

def generate_text(model, vocab, idx_to_char, device, prompt="First", max_len=200, temperature=0.8, top_k=40):
    model.eval()
    prompt_tokens = [vocab.get(c, 0) for c in prompt]
    with torch.no_grad():
        current = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        for _ in range(max_len):
            logits = model(current)
            last_logits = logits[0, -1, :]
            last_logits = last_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(0)))
                last_logits[last_logits < v[-1]] = float('-inf')
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
            if current.size(1) > 512:
                current = current[:, -512:]
    result = ''.join([idx_to_char.get(t, '?') for t in current[0].cpu().tolist()])
    return result

def train_lm(model, train_loader, test_loader, epochs, lr, device, name, vocab=None, idx_to_char=None, weight_decay=0.01, do_inference=True):
    print(f"\n  Training {name}...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0
    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / total
        best_acc = max(best_acc, acc)
        epoch_time = time.time() - epoch_start
        print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, acc={acc:.4f}, time={epoch_time:.1f}s")
        if do_inference and vocab is not None and idx_to_char is not None:
            if epoch in [14, 19] or epoch == epochs - 1:
                print(f"\n    === Generated text after epoch {epoch+1} ===")
                gen_text = generate_text(model, vocab, idx_to_char, device, prompt="First", max_len=300, temperature=0.8, top_k=40)
                print(f"    {gen_text}")
                print(f"    === End of generation ===\n")
    return best_acc

def main():
    print("="*60)
    print("SHAKESPEARE NEXT TOKEN - LONG SEQUENCES (512+)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    text, n_chars = load_shakespeare()
    vocab, vocab_size = build_vocab(text)
    idx_to_char = {i: c for c, i in vocab.items()}
    print(f"Dataset: {n_chars:,} chars, Vocab: {vocab_size}")
    data = np.array([vocab.get(c, 0) for c in text], dtype=np.int64)
    seq_len = 512
    train_start = 0
    train_end = train_start + 50000
    test_start = train_end + seq_len
    test_end = test_start + 5000
    print(f"Train: {train_start:,} - {train_end:,} ({train_end - train_start:,} samples)")
    print(f"Test:  {test_start:,} - {test_end:,} ({test_end - test_start:,} samples)")
    train_ds = ShakespeareSeq2SeqDataset(data[train_start:train_end], seq_len)
    test_ds = ShakespeareSeq2SeqDataset(data[test_start:test_end], seq_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    epochs = 30
    lr = 0.0003
    weight_decay = 0.01
    ed_wat = 40
    n_wat = sum(p.numel() for p in WATModel(vocab_size, ed_wat).parameters())
    ed_trans = 36
    n_trans = sum(p.numel() for p in TransformerLanguageModel(vocab_size, ed_trans).parameters())
    print(f"\nWAT embed={ed_wat}: {n_wat:,} params")
    print(f"Transformer embed={ed_trans}: {n_trans:,} params")
    print(f"Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
    print("\n" + "="*60)
    print("--- Training WAT ---")
    print("="*60)
    wat = WATModel(vocab_size, ed_wat)
    wat_acc = train_lm(wat, train_loader, test_loader, epochs, lr, device, "WAT", 
                       vocab=vocab, idx_to_char=idx_to_char, weight_decay=weight_decay)
    print("\n" + "="*60)
    print("--- Training Transformer  ---")
    print("="*60)
    trans = TransformerLanguageModel(vocab_size, ed_trans)
    trans_acc = train_lm(trans, train_loader, test_loader, epochs, lr, device, "Transformer", 
                         vocab=None, idx_to_char=None, weight_decay=weight_decay, do_inference=False)
    print("\n" + "="*60)
    print("RESULTS - Shakespeare Next Token (512)")
    print("="*60)
    print(f"WAT:         {wat_acc*100:.2f}% ({n_wat:,} params)")
    print(f"Transformer: {trans_acc*100:.2f}% ({n_trans:,} params)")
    print("="*60)

if __name__ == "__main__":
    main()
