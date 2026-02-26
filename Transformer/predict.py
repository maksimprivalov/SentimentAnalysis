import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import DistilBertTokenizerFast

from model import DistilBertSentiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--text", "-t", default=None)
    p.add_argument("texts", nargs="*")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parents[1] / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = Path(args.checkpoint)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name = ckpt["model_name"]
    max_len = ckpt["max_len"]
    freeze_layers = ckpt.get("freeze_layers", 5)
    num_classes = ckpt.get("num_classes", 2)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    model = DistilBertSentiment(
        model_name=model_name,
        num_classes=num_classes,
        freeze_layers=freeze_layers,
    )
    model.load_state_dict(ckpt["state_dict"])
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
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            logits = model(enc["input_ids"], enc["attention_mask"])
            pred = logits.argmax(dim=1).item()
            prob = torch.softmax(logits, dim=1)[0, pred].item()
            print(f"{labels[pred]}\t{prob:.3f}\t{text[:60]}{'...' if len(text) > 60 else ''}")


if __name__ == "__main__":
    main()
