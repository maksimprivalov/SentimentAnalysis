import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np

from model import TextCNN


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return re.findall(r"[a-z0-9]+", text) if lowercase else re.findall(r"\w+", text)


def text_to_ids(text, vocab, max_len):
    unk_id = vocab.get("<unk>", 1)
    toks = tokenize(text, lowercase=True)[:max_len]
    ids = [vocab.get(t, unk_id) for t in toks]
    pad_id = 0
    ids = ids + [pad_id] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--text", "-t", default=None)
    p.add_argument("texts", nargs="*")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parents[1] / ckpt_path
    if not ckpt_path.exists():
        ckpt_path = Path(args.checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    vocab = ckpt["vocab"]
    max_len = ckpt["max_len"]
    emb = ckpt["embedding_weights"]
    if isinstance(emb, torch.Tensor):
        emb = emb.numpy()

    model = TextCNN(
        ckpt["vocab_size"],
        ckpt["embed_dim"],
        emb,
        ckpt["num_filters"],
        ckpt["kernel_sizes"],
        dropout=0.0,
        num_classes=2,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    texts = []
    if args.text:
        texts.append(args.text)
    texts.extend(args.texts)
    if not texts:
        line = sys.stdin.readline()
        if line:
            texts.append(line.strip())
    if not texts:
        print("No text. Use -t '...' or positional args or stdin.", file=sys.stderr)
        sys.exit(1)

    labels = ["negative", "positive"]
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            x = text_to_ids(text.strip(), vocab, max_len)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            prob = torch.softmax(logits, dim=1)[0, pred].item()
            print(f"{labels[pred]}\t{prob:.3f}\t{text[:60]}{'...' if len(text) > 60 else ''}")


if __name__ == "__main__":
    main()
