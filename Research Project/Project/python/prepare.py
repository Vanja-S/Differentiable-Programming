import requests
import re
import os
import torch
import random
import numpy as np

GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
    "https://www.gutenberg.org/files/98/98-0.txt",  # Tale of Two Cities
    "https://www.gutenberg.org/files/345/345-0.txt",  # Dracula
    "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
    "https://www.gutenberg.org/files/174/174-0.txt",  # Dorian Gray
]

ALPHABET_SIZE = 26


def text_to_int(text):
    return [ord(c) - ord("A") for c in text]


def preprocess_text(text):
    text = re.split(
        r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*",
        text,
        flags=re.IGNORECASE,
    )[-1]
    text = re.split(
        r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*",
        text,
        flags=re.IGNORECASE,
    )[0]
    text = text.upper()
    text = re.sub(r"[^A-Z]", "", text)
    return text


def split_samples(text, sample_length):
    samples = []
    for i in range(0, len(text) - sample_length + 1, sample_length):
        sample = text[i : i + sample_length]
        if len(sample) == sample_length:
            samples.append(sample)
    return samples


def coprime_choices(m):
    return [a for a in range(1, m) if np.gcd(a, m) == 1]


def affine_encrypt(plaintext_ints, a, b, m=ALPHABET_SIZE):
    return [(a * x + b) % m for x in plaintext_ints]


def prepare_affine_dataset(
    urls, sample_lengths, out_dir="datasets", train_ratio=0.8, val_ratio=0.1, seed=42
):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    all_samples = {L: [] for L in sample_lengths}
    coprimes = coprime_choices(ALPHABET_SIZE)
    for url in urls:
        print(f"Downloading {url} ...")
        r = requests.get(url)
        r.encoding = "utf-8"
        text = preprocess_text(r.text)
        for L in sample_lengths:
            samples = split_samples(text, L)
            all_samples[L].extend(samples)
    for L, samples in all_samples.items():
        print(f"Preparing {len(samples)} samples of length {L}")
        tuples = []
        for s in samples:
            P = text_to_int(s)
            a = random.choice(coprimes)
            b = random.randint(0, ALPHABET_SIZE - 1)
            C = affine_encrypt(P, a, b)
            tuples.append((C, [a, b], P))
        # Shuffle and split
        random.shuffle(tuples)
        n = len(tuples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        splits = {
            "train": tuples[:n_train],
            "val": tuples[n_train : n_train + n_val],
            "test": tuples[n_train + n_val :],
        }
        for split, data in splits.items():
            C_tensor = torch.tensor([x[0] for x in data], dtype=torch.long)
            K_tensor = torch.tensor([x[1] for x in data], dtype=torch.long)
            P_tensor = torch.tensor([x[2] for x in data], dtype=torch.long)
            torch.save(
                {"ciphertext": C_tensor, "key": K_tensor, "plaintext": P_tensor},
                os.path.join(out_dir, f"affine_{L}_{split}.pt"),
            )
            print(
                f"Saved: {os.path.join(out_dir, f'affine_{L}_{split}.pt')} "
                f"({C_tensor.shape[0]} samples)"
            )
    print("Done.")


if __name__ == "__main__":
    sample_lengths = [100, 500, 1000, 10000]
    prepare_affine_dataset(GUTENBERG_URLS, sample_lengths)
