import argparse
from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Metrics, Model, Output

DEFAULT_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"


@dsl.component(base_image=DEFAULT_IMAGE)
def train_char_lm_op(
    num_gpus: int,
    epochs: int,
    metrics: Output[Metrics],
    model_artifact: Output[Model]
):
    """
    Trains a character-level LSTM language model on domain vocabulary
    and exports model weights and training metrics.
    """
    import os
    import json
    import time
    import torch
    import torch.nn as nn

    print("=" * 65)
    print("STEP 1: Training Character-Level Completion Model")
    print("=" * 65)

    cuda_available = torch.cuda.is_available() and num_gpus > 0
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Executing on hardware device: {device}")

    # 1. Dataset & Vocabulary
    words = [
        "openshift", "kubernetes", "kubeflow", "pipeline", "pytorch",
        "container", "tensor", "artificialintelligence", "machinelearning", "automation"
    ]
    unique_chars = sorted(list(set("".join(words) + "$ ")))
    char2idx = {c: i for i, c in enumerate(unique_chars)}
    idx2char = {i: c for i, c in enumerate(unique_chars)}
    vocab_size = len(char2idx)

    # 2. Model Architecture
    class CharLSTM(nn.Module):
        def __init__(self, vocab_sz, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_sz, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_sz)

        def forward(self, x, hidden=None):
            out = self.embedding(x)
            out, hidden = self.lstm(out, hidden)
            return self.fc(out), hidden

    model = CharLSTM(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    # 3. Training Loop
    start_time = time.time()
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for w in words:
            tokens = [char2idx[c] for c in w + "$"]
            inputs = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(words)

    train_ms = (time.time() - start_time) * 1000

    print(f"[SUCCESS] Training completed in {train_ms:.2f} ms")
    print(f"[METRIC] Final Cross-Entropy Loss: {final_loss:.4f}")
    print(f"[METRIC] Vocabulary Size: {vocab_size} characters")

    # 4. Record Metrics in KFP
    metrics.log_metric("final_loss", round(final_loss, 4))
    metrics.log_metric("epochs", epochs)
    metrics.log_metric("vocab_size", vocab_size)
    metrics.log_metric("train_time_ms", round(train_ms, 2))

    # 5. Save Model and Vocabulary to KFP Model Output Path
    os.makedirs(model_artifact.path, exist_ok=True)
    weights_path = os.path.join(model_artifact.path, "model.pt")
    vocab_path = os.path.join(model_artifact.path, "vocab.json")

    torch.save(model.state_dict(), weights_path)
    with open(vocab_path, "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f, indent=2)

    model_artifact.metadata["framework"] = "PyTorch"
    model_artifact.metadata["model_type"] = "CharLSTM"
    model_artifact.metadata["vocab_size"] = vocab_size

    print(f"[ARTIFACT] Exported model checkpoint to: {weights_path}")
    print(f"[ARTIFACT] Exported vocabulary to: {vocab_path}")
    print("=" * 65)


@dsl.component(base_image=DEFAULT_IMAGE)
def evaluate_completions_op(
    model_artifact: Input[Model],
    evaluation_report: Output[Artifact]
):
    """
    Loads the trained CharLSTM model artifact and runs test word completions
    on prefixes ('ope', 'kube', 'py', 'con', 'ten').
    """
    import os
    import json
    import torch
    import torch.nn as nn

    print("=" * 65)
    print("STEP 2: Evaluating Word Completions from Model Artifact")
    print("=" * 65)

    weights_path = os.path.join(model_artifact.path, "model.pt")
    vocab_path = os.path.join(model_artifact.path, "vocab.json")

    with open(vocab_path) as f:
        vocab_data = json.load(f)
    char2idx = vocab_data["char2idx"]
    idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}
    vocab_size = len(char2idx)

    class CharLSTM(nn.Module):
        def __init__(self, vocab_sz, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_sz, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_sz)

        def forward(self, x, hidden=None):
            embeds = self.embedding(x)
            out, hidden = self.lstm(embeds, hidden)
            return self.fc(out), hidden

    model = CharLSTM(vocab_size)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    def complete(prefix, max_len=20):
        chars_out = list(prefix.lower())
        hidden = None
        with torch.no_grad():
            for c in prefix.lower():
                if c not in char2idx:
                    return prefix
                token = torch.tensor([[char2idx[c]]], dtype=torch.long)
                logits, hidden = model(token, hidden)

            last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            next_c = idx2char[last_token]
            while next_c != "$" and len(chars_out) < max_len:
                chars_out.append(next_c)
                token = torch.tensor([[last_token]], dtype=torch.long)
                logits, hidden = model(token, hidden)
                last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                next_c = idx2char[last_token]
        return "".join(chars_out)

    test_prefixes = ["ope", "kube", "py", "con", "ten", "art", "mac", "auto"]
    results = {}
    print("\nCompletion Evaluation Results:")
    print("-" * 50)
    for p in test_prefixes:
        completion = complete(p)
        results[p] = completion
        print(f"  Input: '{p:<5}' ---> Predicted Word: '{completion}'")
    print("-" * 50)

    # Write evaluation report artifact
    os.makedirs(os.path.dirname(evaluation_report.path), exist_ok=True)
    with open(evaluation_report.path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ARTIFACT] Evaluation report saved to: {evaluation_report.path}")
    print("[SUCCESS] Word completion evaluation finished.")
    print("=" * 65)


@dsl.pipeline(
    name="ocp-char-completion-pipeline",
    description="Trains a PyTorch character-level language model, records metrics, and evaluates text completions."
)
def char_completion_pipeline(num_gpus: int = 0, epochs: int = 150):
    train_task = train_char_lm_op(num_gpus=num_gpus, epochs=epochs)
    train_task.set_caching_options(False)
    train_task.set_cpu_request("500m")
    train_task.set_cpu_limit("2")
    train_task.set_memory_request("1Gi")
    train_task.set_memory_limit("4Gi")

    if num_gpus > 0:
        train_task.set_gpu_limit(num_gpus)
        train_task.set_accelerator_type("nvidia.com/gpu")

    eval_task = evaluate_completions_op(model_artifact=train_task.outputs["model_artifact"])
    eval_task.set_caching_options(False)
    eval_task.set_cpu_request("500m")
    eval_task.set_cpu_limit("1")
    eval_task.set_memory_request("512Mi")
    eval_task.set_memory_limit("2Gi")


def compile_pipeline(output_path: str = "pipeline.yaml"):
    print(f"Compiling pipeline to: {output_path}")
    compiler.Compiler().compile(
        pipeline_func=char_completion_pipeline,
        package_path=output_path
    )
    print("Compilation successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile OpenShift AI Character Completion Pipeline")
    parser.add_argument(
        "--output",
        "-o",
        default="pipeline.yaml",
        help="Path for compiled pipeline YAML"
    )
    args = parser.parse_args()
    compile_pipeline(args.output)
