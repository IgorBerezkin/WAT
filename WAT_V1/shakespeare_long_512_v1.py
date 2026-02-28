import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from WAT_v1 import WATModel

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

class ShakespeareNextTokenLongDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
        self.size = max(0, len(data) - seq_len)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
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
        x = x[:, -1, :]
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

def train_lm(model, train_loader, test_loader, epochs, lr, device, name):
    print(f"\n  Training {name}...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        epoch_time = time.time() - epoch_start
        print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, acc={acc:.4f}, time={epoch_time:.1f}s")
    return best_acc

def main():
    print("="*60)
    print("SHAKESPEARE NEXT TOKEN - LONG SEQUENCES (512+)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    text, n_chars = load_shakespeare()
    vocab, vocab_size = build_vocab(text)
    print(f"Dataset: {n_chars:,} chars, Vocab: {vocab_size}")
    data = np.array([vocab.get(c, 0) for c in text], dtype=np.int64)
    seq_len = 512
    train_start = 0
    train_end = train_start + 50000
    test_start = train_end + seq_len
    test_end = test_start + 5000
    print(f"Train: {train_start:,} - {train_end:,} ({train_end - train_start:,} samples)")
    print(f"Test:  {test_start:,} - {test_end:,} ({test_end - test_start:,} samples)")
    train_ds = ShakespeareNextTokenLongDataset(data[train_start:train_end], seq_len)
    test_ds = ShakespeareNextTokenLongDataset(data[test_start:test_end], seq_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    epochs = 60
    lr = 0.0003
    ed_wat = 40
    n_wat = sum(p.numel() for p in WATModel(vocab_size, ed_wat).parameters())
    ed_trans = 36
    n_trans = sum(p.numel() for p in TransformerLanguageModel(vocab_size, ed_trans).parameters())
    print(f"\nWAT embed={ed_wat}: {n_wat:,} params")
    print(f"Transformer embed={ed_trans}: {n_trans:,} params")
    print(f"Epochs: {epochs}, LR: {lr}")
    print("\n--- Training WAT ---")
    wat = WATModel(vocab_size, ed_wat)
    wat_acc = train_lm(wat, train_loader, test_loader, epochs, lr, device, "WAT")
    print("\n--- Training Transformer ---")
    trans = TransformerLanguageModel(vocab_size, ed_trans)
    trans_acc = train_lm(trans, train_loader, test_loader, epochs, lr, device, "Trans")
    print("\n" + "="*60)
    print("RESULTS - Shakespeare Next Token (512)")
    print("="*60)
    print(f"WAT:         {wat_acc*100:.2f}% ({n_wat:,} params)")
    print(f"Transformer: {trans_acc*100:.2f}% ({n_trans:,} params)")
    print("="*60)

if __name__ == "__main__":
    main()
