import os
import sys
import json
import torch
from model import train_model, complete_word, build_vocab

def main():
    print("=" * 65)
    print("OpenShift AI Pipeline: Character-Level Language Model (CharLSTM)")
    print("=" * 65)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Hardware Device: {device.upper()} (CUDA Available: {cuda_available})")
    
    if cuda_available:
        print(f"Device Name    : {torch.cuda.get_device_name(0)}")
        print(f"Total Memory   : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    
    print("\nTraining Character-Level Completion Model on domain vocabulary...")
    model, char2idx, idx2char, final_loss, train_ms = train_model(
        epochs=150,
        lr=0.02,
        device=device
    )
    
    print(f"[SUCCESS] Training completed in {train_ms:.2f} ms")
    print(f"[METRIC] Final Cross-Entropy Loss: {final_loss:.4f}")
    print(f"[METRIC] Vocabulary Size: {len(char2idx)} characters")
    
    # Run test completions
    test_prefixes = ["ope", "kube", "py", "con", "ten", "art", "mac"]
    print("\n" + "-" * 40)
    print("Testing Interactive Word Completions:")
    print("-" * 40)
    for prefix in test_prefixes:
        completed = complete_word(model, char2idx, idx2char, prefix, device=device)
        print(f"  Prefix: '{prefix:<5}' -> Predicted Completion: '{completed}'")
    print("-" * 40)

    # Save artifacts locally
    output_dir = os.environ.get("MODEL_OUTPUT_DIR", "./artifacts")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.pt")
    vocab_path = os.path.join(output_dir, "vocab.json")
    
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f, indent=2)
        
    print(f"\n[ARTIFACT] Model weights saved to : {model_path}")
    print(f"[ARTIFACT] Vocabulary saved to    : {vocab_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
