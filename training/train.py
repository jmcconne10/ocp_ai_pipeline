#!/usr/bin/env python3
"""
Model Training Script
Trains a CharLSTM completion model on configurable domain vocabulary words
and exports model weights and vocabulary mappings.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add current dir to import model.py
sys.path.insert(0, str(Path(__file__).parent))
from model import CharLSTM, complete_word, train_model, build_vocab


def load_words(words_file: str = None, words_arg: str = None) -> list:
    words = []
    # 1. From CLI comma-separated argument
    if words_arg:
        words = [w.strip().lower() for w in words_arg.split(",") if w.strip()]
        if words:
            print(f"Loaded {len(words)} training words from CLI argument.")
            return words

    # 2. From file
    if words_file and os.path.exists(words_file):
        with open(words_file) as f:
            words = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
        if words:
            print(f"Loaded {len(words)} training words from file: {words_file}")
            return words

    # 3. Default fallback
    default_file = Path(__file__).parent.parent / "data" / "words.txt"
    if default_file.exists():
        with open(default_file) as f:
            words = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
        if words:
            print(f"Loaded {len(words)} training words from default data/words.txt")
            return words

    from model import DEFAULT_WORDS
    print("Using built-in default vocabulary words.")
    return DEFAULT_WORDS


def main():
    parser = argparse.ArgumentParser(description="Train Character-Level Language Model")
    parser.add_argument("--words-file", "-f", default=None, help="Path to text file containing training words")
    parser.add_argument("--words", "-w", default=None, help="Comma-separated list of words to train on")
    parser.add_argument("--epochs", "-e", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate (default: 0.02)")
    parser.add_argument("--output-dir", "-o", default="artifacts", help="Directory to save model.pt and vocab.json")
    parser.add_argument("--device", "-d", default=None, help="Hardware device: 'cpu' or 'cuda'")

    args = parser.parse_args()

    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("Training Character-Level Completion Model (CharLSTM)")
    print("=" * 65)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"Compute Device  : {device.upper()}")

    words = load_words(args.words_file, args.words)
    print(f"Vocabulary Words ({len(words)}): {', '.join(words[:8])}{'...' if len(words) > 8 else ''}")

    model, char2idx, idx2char, final_loss, train_ms = train_model(
        words=words,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    print(f"\n[SUCCESS] Training completed in {train_ms:.2f} ms")
    print(f"[METRIC] Final Cross-Entropy Loss: {final_loss:.4f}")
    print(f"[METRIC] Unique Characters: {len(char2idx)}")

    # Test sample completions
    sample_prefixes = [w[:min(3, len(w))] for w in words[:6]]
    print("\nSample In-Memory Completions:")
    print("-" * 45)
    for p in sample_prefixes:
        comp = complete_word(model, char2idx, idx2char, p, device=device)
        print(f"  Input: '{p:<5}'  --->  Completed: '{comp}'")
    print("-" * 45)

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, "model.pt")
    vocab_path = os.path.join(args.output_dir, "vocab.json")

    torch.save(model.state_dict(), weights_path)
    with open(vocab_path, "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f, indent=2)

    print(f"\n[ARTIFACT] Model checkpoint saved : {weights_path}")
    print(f"[ARTIFACT] Vocabulary file saved   : {vocab_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
