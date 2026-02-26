import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import random
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import TextCNN


def _parse_row(row):
    if len(row) < 3:
        return None, None
    label_s, body = row[0].strip(), row[2].strip()
    if label_s.startswith('"') and label_s.endswith('"'):
        label_s = label_s[1:-1].replace('""', '"')
    if body.startswith('"') and body.endswith('"'):
        body = body[1:-1].replace('""', '"')
    if label_s not in ("1", "2"):
        return None, None
    label = 0 if label_s == "1" else 1
    if not body.strip():
        return None, None
    return body.strip(), label


def load_csv(path, max_rows=None):
    path = Path(path)
    texts, labels = [], []
    with open(path, "r", encoding="utf-8", newline="", errors="replace") as f:
        for row in csv.reader(f):
            if max_rows is not None and len(texts) >= max_rows:
                break
            text, label = _parse_row(row)
            if text is not None:
                texts.append(text)
                labels.append(label)
    return texts, labels


def load_csv_balanced(path, max_negative, max_positive, seed=42):
    # we take n positive and n negative reviews so we can balance dataset
    path = Path(path)
    neg_texts, pos_texts = [], []
    with open(path, "r", encoding="utf-8", newline="", errors="replace") as f:
        for row in csv.reader(f):
            if len(neg_texts) >= max_negative and len(pos_texts) >= max_positive:
                break
            text, label = _parse_row(row)
            if text is None:
                continue
            if label == 0:
                if len(neg_texts) < max_negative:
                    neg_texts.append(text)
            else:
                if len(pos_texts) < max_positive:
                    pos_texts.append(text)
    texts = neg_texts + pos_texts
    labels = [0] * len(neg_texts) + [1] * len(pos_texts)
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    return texts, labels

#words processing
def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return re.findall(r"[a-z0-9]+", text) if lowercase else re.findall(r"\w+", text)

def build_vocab(texts, tokenizer, max_size=None):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenizer(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(max_size) if max_size else cnt.most_common():
        if w not in vocab:
            vocab[w] = len(vocab)
            if max_size and len(vocab) >= max_size:
                break
    return vocab


# loading vectorse from GloVe
def load_glove(path):
    path = Path(path)
    word2vec = {}
    dim = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            word, vec = parts[0], parts[1:]
            try:
                vec = np.array([float(x) for x in vec], dtype=np.float32)
            except ValueError:
                continue
            if dim is None:
                dim = len(vec)
            elif len(vec) != dim:
                continue
            word2vec[word] = vec
    return word2vec, dim or 0

# matrix of weights for words, if have word in GloVe we use it, else we randomize
def build_embedding_matrix(vocab, word2vec, dim):
    V = len(vocab)
    matrix = np.random.randn(V, dim).astype(np.float32) * 0.05
    for w, idx in vocab.items():
        if w in ("<pad>", "<unk>"):
            continue
        if w in word2vec:
            matrix[idx] = word2vec[w]
    return matrix


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = vocab.get("<unk>", 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        toks = self.tokenizer(self.texts[i])[: self.max_len]
        ids = [self.vocab.get(t, self.unk_id) for t in toks]
        return ids, self.labels[i]

#making same length reviews for batch
def collate(batch):
    pad_id = 0
    max_len = max(len(x[0]) for x in batch)
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids, _ in batch]
    labels = [lab for _, lab in batch]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # обнуляем градиенты
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        # считаем как изменить веса используя производную
        loss.backward()
        # обновляем веса
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        n += inputs.size(0)
    return total_loss / n if n else 0.0


#standard evaluation
def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad(): 
        for inputs, labels in loader:
            out = model(inputs.to(device)).argmax(dim=1).cpu().numpy()
            preds.extend(out)
            true.extend(labels.numpy())
    return {"accuracy": accuracy_score(true, preds), "f1": f1_score(true, preds, average="macro")}


def main():
    #so we can change parameters from command line (i use it for tests)
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="Amazon_Reviews")
    p.add_argument("--glove_path", default="wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_test", type=int, default=None)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--max_per_class_test", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--save", default="../outputs/checkpoints/model.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.quick:
        args.max_per_class = args.max_per_class or 15_000
        args.max_per_class_test = args.max_per_class_test or 3_250
        args.epochs = 3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # at the beginning i thoug that we will use collegue's computer for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir

    #at the begingng i didnt thought that first n rows are not balanced
    if args.max_per_class is not None:
        train_texts, train_labels = load_csv_balanced(
            data_dir / "train.csv",
            max_negative=args.max_per_class,
            max_positive=args.max_per_class,
            seed=args.seed,
        )
        print(f"Train (balanced): {len(train_texts)}")
    else:
        train_texts, train_labels = load_csv(data_dir / "train.csv", max_rows=args.max_train)
        print(f"Train: {len(train_texts)}")

    #at the begingng i didnt thought that first n rows are not balanced
    if args.max_per_class_test is not None:
        test_texts, test_labels = load_csv_balanced(
            data_dir / "test.csv",
            max_negative=args.max_per_class_test,
            max_positive=args.max_per_class_test,
            seed=args.seed + 1,
        )
        print(f"Test (balanced): {len(test_texts)}")
    else:
        test_texts, test_labels = load_csv(data_dir / "test.csv", max_rows=args.max_test)
        print(f"Test: {len(test_texts)}")

    #to lower case
    tokenizer = lambda t: tokenize(t, lowercase=True)
    # take only 100k most popular words
    vocab = build_vocab(train_texts, tokenizer, max_size=100_000)
    print(f"Vocab size: {len(vocab)}")

    glove_path = root / args.glove_path if not Path(args.glove_path).is_absolute() else Path(args.glove_path)
    # vector + dimension
    word2vec, dim = load_glove(glove_path)
    print(f"GloVe: {len(word2vec)} words, dim={dim}")

    emb_matrix = build_embedding_matrix(vocab, word2vec, dim)

    train_ds = ReviewDataset(train_texts, train_labels, tokenizer, vocab, args.max_len)
    test_ds = ReviewDataset(test_texts, test_labels, tokenizer, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    kernel_sizes = (3, 4, 5)
    model = TextCNN(len(vocab), dim, emb_matrix, 100, kernel_sizes, 0.5, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, criterion)
        m = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}  loss={loss:.4f}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
    print(f"Training time: {time.perf_counter() - start:.1f} s")

    m = evaluate(model, test_loader, device)
    print(f"Final  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    save_path = root / "CNN" / args.save if not Path(args.save).is_absolute() else Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "max_len": args.max_len,
        "embed_dim": dim,
        "num_filters": 100,
        "kernel_sizes": list(kernel_sizes),
        "vocab_size": len(vocab),
        "embedding_weights": emb_matrix,
    }, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
