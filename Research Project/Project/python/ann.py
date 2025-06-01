import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json


class ModularBranch(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(26, 16)
        self.fc1 = nn.Linear(seq_len * 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class StatisticalBranch(nn.Module):
    def __init__(self, stat_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(stat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class HybridAffineNet(nn.Module):
    def __init__(self, seq_len, stat_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.modular = ModularBranch(seq_len, hidden_dim)
        self.stat = StatisticalBranch(stat_dim, hidden_dim)
        self.fc_final = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, x_cipher, x_stat):
        mod_out = self.modular(x_cipher)
        stat_out = self.stat(x_stat)
        x = torch.cat([mod_out, stat_out], dim=1)
        return self.fc_final(x)


def letter_freqs(batch, alphabet_size=26):
    """
    Calculates the frequency of each letter in a batch of sequences.
    Args:
      batch (torch.Tensor): A tensor of shape (batch_size, seq_len) containing integer-encoded sequences, where each value represents a letter.
      alphabet_size (int, optional): The number of unique letters in the alphabet. Defaults to 26.
    Returns:
      torch.Tensor: A tensor of shape (batch_size, alphabet_size) where each row contains the normalized frequency of each letter in the corresponding sequence.
    """
    batch_size, seq_len = batch.shape
    freqs = torch.zeros((batch_size, alphabet_size), dtype=torch.float32)
    for i in range(batch_size):
        vals, counts = torch.unique(batch[i], return_counts=True)
        freqs[i, vals] = counts.float()
    freqs /= batch.shape[1]
    return freqs


def load_dataset(path):
    data = torch.load(path)
    return data["ciphertext"], data["key"], data["plaintext"]


import json


def train_and_evaluate(
    train_path,
    val_path,
    test_path,
    seq_len,
    stat_dim,
    hidden_dim=128,
    batch_size=128,
    lr=1e-3,
    epochs=10,
    device="cpu",
    results_prefix="results_affine",
):
    # Load data
    X_train, K_train, _ = load_dataset(train_path)
    X_val, K_val, _ = load_dataset(val_path)
    X_test, K_test, _ = load_dataset(test_path)

    # Model, loss, optimizer
    model = HybridAffineNet(seq_len, stat_dim, hidden_dim, out_dim=312).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare key labels for classification: map (a, b) to a single class index
    coprimes = [a for a in range(1, 26) if np.gcd(a, 26) == 1]
    key_to_idx = {
        (a, b): i
        for i, (a, b) in enumerate((a, b) for a in coprimes for b in range(26))
    }
    idx_to_key = {i: (a, b) for (a, b), i in key_to_idx.items()}

    def keys_to_class_idx(keys):
        return torch.tensor(
            [key_to_idx[(int(a), int(b))] for a, b in keys], dtype=torch.long
        )

    def make_batches(X, K, batch_size):
        n = X.shape[0]
        for i in range(0, n, batch_size):
            yield X[i : i + batch_size], K[i : i + batch_size]

    # Track accuracy curves
    train_acc_curve = []
    val_acc_curve = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total = 0
        for Xb, Kb in tqdm(
            make_batches(X_train, K_train, batch_size), desc=f"Epoch {epoch+1}"
        ):
            Xb, Kb = Xb.to(device), Kb.to(device)
            stat_feats = letter_freqs(Xb).to(device)
            key_labels = keys_to_class_idx(Kb).to(device)
            optimizer.zero_grad()
            logits = model(Xb, stat_feats)
            loss = criterion(logits, key_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == key_labels).sum().item()
            total += Xb.size(0)
        train_acc = total_correct / total
        train_acc_curve.append(train_acc)
        print(f"Train loss: {total_loss/total:.4f}, Train acc: {train_acc:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total = 0
            for Xb, Kb in make_batches(X_val, K_val, batch_size):
                Xb, Kb = Xb.to(device), Kb.to(device)
                stat_feats = letter_freqs(Xb).to(device)
                key_labels = keys_to_class_idx(Kb).to(device)
                logits = model(Xb, stat_feats)
                preds = logits.argmax(dim=1)
                total_correct += (preds == key_labels).sum().item()
                total += Xb.size(0)
            val_acc = total_correct / total
            val_acc_curve.append(val_acc)
            print(f"Validation acc: {val_acc:.4f}")

    # Test set evaluation
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total = 0
        for Xb, Kb in make_batches(X_test, K_test, batch_size):
            Xb, Kb = Xb.to(device), Kb.to(device)
            stat_feats = letter_freqs(Xb).to(device)
            key_labels = keys_to_class_idx(Kb).to(device)
            logits = model(Xb, stat_feats)
            preds = logits.argmax(dim=1)
            total_correct += (preds == key_labels).sum().item()
            total += Xb.size(0)
        test_acc = total_correct / total
        print(f"Test acc: {test_acc:.4f}")

    results_dict = {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "test_accuracy": test_acc,
        "train_acc_curve": train_acc_curve,
        "val_acc_curve": val_acc_curve,
    }
    filename = f"{results_prefix}_{seq_len}.json"
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {filename}")

    return model, test_acc


def run_all_sample_sizes(
    sample_sizes=[100, 500, 1000, 10000],
    hidden_dim=128,
    batch_size=128,
    lr=1e-3,
    epochs=30,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    stat_dim = 26  # Letter frequencies

    results = []
    for seq_len in sample_sizes:
        print(f"\n=== Training for sample size {seq_len} ===")
        train_path = f"datasets/affine_{seq_len}_train.pt"
        val_path = f"datasets/affine_{seq_len}_val.pt"
        test_path = f"datasets/affine_{seq_len}_test.pt"

        model, test_acc = train_and_evaluate(
            train_path,
            val_path,
            test_path,
            seq_len,
            stat_dim,
            hidden_dim,
            batch_size,
            lr,
            epochs,
            device,
        )
        results.append(
            {
                "Sample Length": seq_len,
                "Hidden Dim": hidden_dim,
                "Batch Size": batch_size,
                "Learning Rate": lr,
                "Epochs": epochs,
                "Test Accuracy": test_acc,
            }
        )

    print("\n=== Summary Table ===")
    print(
        f"{'Sample Len':>10} | {'Hidden':>6} | {'Batch':>5} | {'LR':>6} | {'Epochs':>6} | {'Test Acc (%)':>12}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['Sample Length']:>10} | {r['Hidden Dim']:>6} | {r['Batch Size']:>5} | {r['Learning Rate']:.0e} | {r['Epochs']:>6} | {r['Test Accuracy']*100:>11.2f}"
        )

    return results


if __name__ == "__main__":
    run_all_sample_sizes()
