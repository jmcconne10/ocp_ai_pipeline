#!/usr/bin/env python3
"""
Interactive Character-Level Word Completion Inference Script
Loads the trained CharLSTM model. If not found locally, it automatically
downloads the latest trained model.pt artifact from your OpenShift AI cluster.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to python path for model imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    import torch.nn as nn
    from model import CharLSTM, complete_word, train_model, build_vocab
except ImportError:
    print("❌ Error: PyTorch is required to run inference.")
    print("Install dependencies with: pip install torch")
    sys.exit(1)

# Load environment configuration (.env)
load_dotenv(override=False)


def download_from_cluster(output_dir: str = "artifacts") -> bool:
    """
    Downloads the latest model.pt and vocab.json artifacts from the
    OpenShift AI pipeline storage (MinIO) into the local output_dir.
    """
    namespace = os.getenv("NAMESPACE", "").strip()
    if not namespace:
        try:
            res = subprocess.run(["oc", "project", "-q"], capture_output=True, text=True, check=True)
            namespace = res.stdout.strip()
        except Exception:
            return False

    print(f"Connecting to OpenShift cluster namespace '{namespace}'...")
    try:
        # Find minio pod
        pod_res = subprocess.run(
            ["oc", "get", "pods", "-l", "app=minio-dspa", "-n", namespace, "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True, text=True, check=True
        )
        pod = pod_res.stdout.strip()
        if not pod:
            # Fallback to standalone minio label if applicable
            pod_res = subprocess.run(
                ["oc", "get", "pods", "-l", "app=minio", "-n", namespace, "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True, text=True, check=True
            )
            pod = pod_res.stdout.strip()

        if not pod:
            print("  [INFO] MinIO storage pod not found in namespace.")
            return False

        # Find latest model.pt in pipeline storage
        find_res = subprocess.run(
            ["oc", "exec", pod, "-n", namespace, "--", "find", "/data/mlpipeline", "-name", "model.pt"],
            capture_output=True, text=True, check=True
        )
        paths = [p for p in find_res.stdout.splitlines() if p.startswith("/data/mlpipeline")]
        if not paths:
            print("  [INFO] No model.pt artifacts found in cluster pipeline storage yet.")
            return False

        latest_model = paths[-1]
        latest_vocab = latest_model.replace("model.pt", "vocab.json")

        # Extract Run ID from path for informative logging
        parts = latest_model.split("/")
        run_id = parts[3] if len(parts) > 3 else "unknown"
        print(f"Found latest cluster pipeline model from run: {run_id}")

        os.makedirs(output_dir, exist_ok=True)
        weights_path = Path(output_dir) / "model.pt"
        vocab_path = Path(output_dir) / "vocab.json"

        print(f"Downloading model.pt from cluster ({pod})...")
        with open(weights_path, "wb") as f:
            subprocess.run(["oc", "exec", pod, "-n", namespace, "--", "cat", latest_model], stdout=f, check=True)

        print("Downloading vocab.json from cluster...")
        with open(vocab_path, "wb") as f:
            subprocess.run(["oc", "exec", pod, "-n", namespace, "--", "cat", latest_vocab], stdout=f, check=True)

        print(f"✅ Successfully downloaded cluster model artifacts to '{output_dir}/'")
        return True

    except Exception as e:
        print(f"  [WARNING] Could not download model from cluster: {e}")
        return False


def load_or_train_model(model_dir: str = "artifacts", force_cluster_download: bool = False):
    weights_path = Path(model_dir) / "model.pt"
    vocab_path = Path(model_dir) / "vocab.json"

    # If missing or explicitly requested, download from OpenShift cluster
    if force_cluster_download or not (weights_path.exists() and vocab_path.exists()):
        print(f"Model artifacts not found locally in './{model_dir}'.")
        print("Attempting to download latest trained model from OpenShift AI pipeline...")
        downloaded = download_from_cluster(model_dir)
        if not downloaded:
            print("⚠️ Falling back to quick local training (takes ~1 second)...")
            model, char2idx, idx2char, loss, train_ms = train_model(epochs=150, device="cpu")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            with open(vocab_path, "w") as f:
                json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f, indent=2)
            print(f"✅ Local model saved to {weights_path}")
            return model, char2idx, idx2char

    print(f"Loading model checkpoint from: {weights_path}")
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    char2idx = vocab_data["char2idx"]
    idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}
    vocab_size = len(char2idx)

    model = CharLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model, char2idx, idx2char


def run_interactive(model, char2idx, idx2char):
    print("\n" + "=" * 60)
    print("🤖 Interactive AI Word Completion")
    print("Type a prefix (e.g. 'ope', 'kube', 'py', 'ten') and press Enter.")
    print("Type 'exit' or press Ctrl+C to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            prefix = input("Prompt > ").strip()
            if not prefix:
                continue
            if prefix.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            
            completion = complete_word(model, char2idx, idx2char, prefix, device="cpu")
            print(f"  └── Predicted completion: '{completion}'\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Run AI Word Completion Inference")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Input character prefix to complete (e.g., 'ope', 'kube', 'py')"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive terminal completion prompt"
    )
    parser.add_argument(
        "--model-dir",
        "-m",
        default="artifacts",
        help="Directory containing model.pt and vocab.json (default: artifacts)"
    )
    parser.add_argument(
        "--from-cluster",
        "-c",
        action="store_true",
        help="Force re-downloading the latest model artifact from the OpenShift AI cluster"
    )
    args = parser.parse_args()

    model, char2idx, idx2char = load_or_train_model(
        model_dir=args.model_dir,
        force_cluster_download=args.from_cluster
    )

    if args.interactive or not args.prompt:
        if not args.prompt:
            test_prefixes = ["ope", "kube", "py", "con", "ten", "art", "mac"]
            print("\nSample Model Predictions:")
            print("-" * 45)
            for p in test_prefixes:
                pred = complete_word(model, char2idx, idx2char, p, device="cpu")
                print(f"  Input: '{p:<5}'  --->  Completed: '{pred}'")
            print("-" * 45)
            run_interactive(model, char2idx, idx2char)
        else:
            run_interactive(model, char2idx, idx2char)
    else:
        result = complete_word(model, char2idx, idx2char, args.prompt, device="cpu")
        print("\n" + "=" * 45)
        print(f"Input Prefix  : '{args.prompt}'")
        print(f"Predicted Word: '{result}'")
        print("=" * 45)


if __name__ == "__main__":
    main()
