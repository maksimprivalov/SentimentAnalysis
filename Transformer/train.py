import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from model import DistilBertSentiment


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


def load_csv_balanced(path, max_negative, max_positive, seed=42):
    # take n positive and n negative reviews so the dataset is balanced
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
    return [texts[i] for i in idx], [labels[i] for i in idx]


class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_len):
    """Tokenizes a batch of (text, label) pairs into tensors."""
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)
    return collate


def train_epoch(model, loader, optimizer, scheduler, device, criterion, grad_accum):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()
    for step, (input_ids, attn_mask, labels) in enumerate(loader):
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        loss = criterion(model(input_ids, attn_mask), labels) / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            # gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum * labels.size(0)
        n += labels.size(0)
    return total_loss / n if n else 0.0


def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            logits = model(input_ids.to(device), attn_mask.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            true.extend(labels.numpy())
    return {
        "accuracy": accuracy_score(true, preds),
        "f1": f1_score(true, preds, average="macro"),
    }


def main():
    # Argument parsing with defaults and help messages for each parameter, prepared with ai
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="Amazon_Reviews")
    p.add_argument("--model_name", default="distilbert-base-uncased",
                   help="HuggingFace model name. DistilBERT is used by default for speed.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128,
                   help="Max token length. 128 is 4x faster than 512 with minimal accuracy loss.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--freeze_layers", type=int, default=5,
                   help="Freeze first N of 6 DistilBERT layers. 5 = only last block trains.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max_per_class", type=int, default=10_000)
    p.add_argument("--max_per_class_test", type=int, default=3_000)
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: 4k/class, 2 epochs (~25-35 min on CPU)")
    p.add_argument("--save", default="../outputs/checkpoints/transformer_model.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.quick:
        args.max_per_class = args.max_per_class or 4_000
        args.max_per_class_test = args.max_per_class_test or 1_000
        args.epochs = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir

    print("Loading data...")
    train_texts, train_labels = load_csv_balanced(
        data_dir / "train.csv",
        max_negative=args.max_per_class,
        max_positive=args.max_per_class,
        seed=args.seed,
    )
    test_texts, test_labels = load_csv_balanced(
        data_dir / "test.csv",
        max_negative=args.max_per_class_test,
        max_positive=args.max_per_class_test,
        seed=args.seed + 1,
    )
    print(f"Train (balanced): {len(train_texts)}")
    print(f"Test  (balanced): {len(test_texts)}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    train_ds = ReviewDataset(train_texts, train_labels)
    test_ds = ReviewDataset(test_texts, test_labels)
    collate_fn = make_collate(tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print(f"Loading model: {args.model_name} (freeze_layers={args.freeze_layers})")
    model = DistilBertSentiment(
        model_name=args.model_name,
        freeze_layers=args.freeze_layers,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # AdamW is standard for fine-tuning transformers
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    num_steps = len(train_loader) * args.epochs // max(args.grad_accum, 1)
    warmup_steps = num_steps // 10
    # linear warmup + decay is the standard transformer learning rate schedule
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nStarting training: {args.epochs} epochs x {len(train_loader)} steps")
    print(f"Estimated time on CPU: ~{len(train_loader) * args.epochs * 3 // 60} min\n")

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, args.grad_accum)
        m = evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - start
        print(f"Epoch {epoch}  loss={loss:.4f}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  time={elapsed:.0f}s")

    total_time = time.perf_counter() - start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    m = evaluate(model, test_loader, device)
    print(f"Final  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    save_path = (root / "Transformer" / args.save
                 if not Path(args.save).is_absolute() else Path(args.save))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": args.model_name,
        "max_len": args.max_len,
        "freeze_layers": args.freeze_layers,
        "num_classes": 2,
    }, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
