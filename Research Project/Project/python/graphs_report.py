import json
import matplotlib.pyplot as plt

# List your result files here
result_files = [
    "results_affine_100.json",
    "results_affine_500.json",
    "results_affine_1000.json",
    "results_affine_10000.json",
]

import matplotlib.pyplot as plt
import json

result_files = [
    "results_affine_100.json",
    "results_affine_500.json",
    "results_affine_1000.json",
    "results_affine_10000.json",
]

seq_lens = []
test_accs = []
for fname in result_files:
    with open(fname, "r") as f:
        res = json.load(f)
        seq_lens.append(res["seq_len"])
        test_accs.append(res["test_accuracy"] * 100)  # percent

plt.figure(figsize=(7, 5))
plt.plot(seq_lens, test_accs, marker="o", linewidth=2)
plt.xscale("log")
plt.title("Affine Cipher Key Recovery: Test Accuracy vs. Ciphertext Length")
plt.xlabel("Ciphertext Length (log scale)")
plt.ylabel("Test Accuracy (%)")
plt.xticks(seq_lens, labels=[str(x) for x in seq_lens])
plt.ylim(0, 105)
plt.grid(True, which="both", axis="x")
plt.tight_layout()
plt.savefig("affine_accuracy_vs_length.png", format="png")


sample_sizes = [100, 500, 1000, 10000]
for size in sample_sizes:
    fname = f"results_affine_{size}.json"
    with open(fname) as f:
        res = json.load(f)
    epochs = list(range(1, len(res["train_acc_curve"]) + 1))
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [a * 100 for a in res["train_acc_curve"]], label="Train")
    plt.plot(epochs, [a * 100 for a in res["val_acc_curve"]], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Learning Curve (Ciphertext Length {size})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curve_{size}.png", format="png")
