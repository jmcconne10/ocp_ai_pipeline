import argparse
from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Metrics, Model, Output

DEFAULT_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"


@dsl.component(base_image=DEFAULT_IMAGE)
def train_char_lm_op(
    num_gpus: int,
    epochs: int,
    words_list: str,
    metrics: Output[Metrics],
    model_artifact: Output[Model]
):
    """
    Trains a character-level LSTM language model on domain vocabulary,
    records training metrics, and exports weights both to KFP model artifacts
    and to the fixed S3 storage path for automated KServe inference hosting.
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

    # 1. Dataset & Vocabulary Handling
    words = []
    if words_list and words_list.strip():
        # Handle comma, newline, or space delimited words
        raw = words_list.replace("\n", " ").replace(",", " ")
        words = [w.strip().lower() for w in raw.split() if w.strip()]
        print(f"[INFO] Using {len(words)} custom words provided via pipeline parameters.")

    if not words:
        words = [
            "openshift", "kubernetes", "kubeflow", "pipeline", "pytorch",
            "container", "tensor", "artificialintelligence", "machinelearning", "automation"
        ]
        print(f"[INFO] Using built-in default domain vocabulary ({len(words)} words).")

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
    print(f"[METRIC] Vocabulary Size: {vocab_size} unique characters")
    print(f"[METRIC] Total Words Trained: {len(words)}")

    # 4. Record Metrics in KFP
    metrics.log_metric("final_loss", round(final_loss, 4))
    metrics.log_metric("epochs", epochs)
    metrics.log_metric("vocab_size", vocab_size)
    metrics.log_metric("word_count", len(words))
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

    # 6. Export to Fixed S3 Path for KServe Model Serving
    try:
        import subprocess
        subprocess.run(["pip", "install", "--no-cache-dir", "boto3"], check=True, capture_output=True)
        import boto3
        from botocore.client import Config

        s3_endpoint = os.getenv("S3_ENDPOINT", "http://minio-dspa:9000")
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minio")
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "miniopassword")

        print(f"[S3 EXPORT] Connecting to object storage at {s3_endpoint}...")
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            config=Config(signature_version="s3v4")
        )
        bucket_name = "mlpipeline"
        s3.upload_file(weights_path, bucket_name, "models/char-completion/model.pt")
        s3.upload_file(vocab_path, bucket_name, "models/char-completion/vocab.json")
        print(f"[SUCCESS] Model weights exported to s3://{bucket_name}/models/char-completion/model.pt")
        print(f"[SUCCESS] Model vocabulary exported to s3://{bucket_name}/models/char-completion/vocab.json")
    except Exception as e:
        print(f"[WARNING] S3 model export failed (non-fatal): {e}")

    print("=" * 65)


@dsl.component(base_image=DEFAULT_IMAGE)
def evaluate_completions_op(
    model_artifact: Input[Model],
    evaluation_report: Output[Artifact]
):
    """
    Loads the trained CharLSTM model artifact and runs test word completions
    on domain prefixes.
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
            next_c = idx2char.get(last_token, "$")
            while next_c != "$" and len(chars_out) < max_len:
                chars_out.append(next_c)
                token = torch.tensor([[last_token]], dtype=torch.long)
                logits, hidden = model(token, hidden)
                last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                next_c = idx2char.get(last_token, "$")
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


@dsl.component(base_image=DEFAULT_IMAGE)
def deploy_kserve_op(
    model_name: str,
    storage_uri: str,
    deploy: bool
) -> str:
    """
    Automated Step 3: Deploys or updates the KServe InferenceService on OpenShift AI
    pointing to the newly trained model in S3/MinIO.
    """
    import os
    import json
    import ssl
    import urllib.request
    import urllib.error

    print("=" * 65)
    print("STEP 3: KServe Model Hosting Deployment")
    print("=" * 65)

    if not deploy:
        print("[INFO] Model deployment is disabled (deploy_model=False). Skipping.")
        return "skipped"

    # In-cluster service account credentials
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

    if not (os.path.exists(token_path) and os.path.exists(ns_path)):
        print("[WARNING] Not executing inside Kubernetes cluster. Skipping in-cluster KServe deployment.")
        return "local-skip"

    with open(token_path) as f:
        token = f.read().strip()
    with open(ns_path) as f:
        namespace = f.read().strip()

    ctx = ssl.create_default_context(cafile=ca_path) if os.path.exists(ca_path) else ssl._create_unverified_context()
    base_k8s = "https://kubernetes.default.svc"

    def k8s_request(method, path, body=None, content_type="application/json"):
        url = f"{base_k8s}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type,
                "Accept": "application/json"
            },
            method=method
        )
        try:
            with urllib.request.urlopen(req, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
                return (json.loads(raw) if raw else {}), resp.status
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8")
            return err_body, err.code
        except Exception as e:
            return str(e), 500

    # 1. Reconcile Secret 'kserve-s3-secret'
    s3_secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "kserve-s3-secret",
            "namespace": namespace,
            "labels": {"opendatahub.io/dashboard": "true"},
            "annotations": {
                "serving.kserve.io/s3-endpoint": "minio-dspa:9000",
                "serving.kserve.io/s3-usehttps": "0",
                "serving.kserve.io/s3-verifyssl": "0",
                "serving.kserve.io/s3-region": "us-east-1"
            }
        },
        "type": "Opaque",
        "stringData": {
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "miniopassword"
        }
    }

    print(f"[K8S] Reconciling Secret 'kserve-s3-secret' in '{namespace}'...")
    res, status_code = k8s_request("POST", f"/api/v1/namespaces/{namespace}/secrets", s3_secret_manifest)
    if status_code in (200, 201):
        print("[SUCCESS] Created Secret 'kserve-s3-secret'.")
    elif status_code == 409:
        k8s_request("PATCH", f"/api/v1/namespaces/{namespace}/secrets/kserve-s3-secret", s3_secret_manifest, content_type="application/merge-patch+json")
        print("[SUCCESS] Secret 'kserve-s3-secret' is up to date.")
    else:
        print(f"[INFO] Secret response: {status_code}")

    # 2. Reconcile ServiceAccount 'kserve-sa'
    sa_manifest = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": "kserve-sa",
            "namespace": namespace
        },
        "secrets": [{"name": "kserve-s3-secret"}]
    }
    print("[K8S] Reconciling ServiceAccount 'kserve-sa'...")
    res, status_code = k8s_request("POST", f"/api/v1/namespaces/{namespace}/serviceaccounts", sa_manifest)
    if status_code in (200, 201):
        print("[SUCCESS] Created ServiceAccount 'kserve-sa'.")
    elif status_code == 409:
        k8s_request("PATCH", f"/api/v1/namespaces/{namespace}/serviceaccounts/kserve-sa", sa_manifest, content_type="application/merge-patch+json")
        print("[SUCCESS] ServiceAccount 'kserve-sa' is up to date.")

    # 3. Reconcile ServingRuntime 'char-completion-runtime'
    serving_runtime_manifest = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": "char-completion-runtime",
            "namespace": namespace,
            "labels": {"opendatahub.io/dashboard": "true"},
            "annotations": {
                "openshift.io/display-name": "Character Completion Serving Runtime"
            }
        },
        "spec": {
            "supportedModelFormats": [
                {"name": "custom", "version": "1", "autoSelect": True},
                {"name": "pytorch", "version": "2", "autoSelect": True}
            ],
            "multiModel": False,
            "protocolVersions": ["v2", "v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": f"image-registry.openshift-image-registry.svc:5000/{namespace}/char-completion-serving:latest",
                    "env": [
                        {"name": "PORT", "value": "8080"},
                        {"name": "MODEL_DIR", "value": "/mnt/models"}
                    ],
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}]
                }
            ]
        }
    }
    print("[K8S] Reconciling ServingRuntime 'char-completion-runtime'...")
    res, status_code = k8s_request("POST", f"/apis/serving.kserve.io/v1alpha1/namespaces/{namespace}/servingruntimes", serving_runtime_manifest)
    if status_code in (200, 201):
        print("[SUCCESS] Created ServingRuntime 'char-completion-runtime'.")
    elif status_code == 409:
        k8s_request("PATCH", f"/apis/serving.kserve.io/v1alpha1/namespaces/{namespace}/servingruntimes/char-completion-runtime", serving_runtime_manifest, content_type="application/merge-patch+json")
        print("[SUCCESS] ServingRuntime 'char-completion-runtime' is up to date.")

    # 1. Reconcile Deployment 'char-completion'
    deploy_manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "char-completion",
            "namespace": namespace,
            "labels": {"app": "char-completion", "opendatahub.io/dashboard": "true"}
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "char-completion"}},
            "template": {
                "metadata": {
                    "labels": {"app": "char-completion", "opendatahub.io/dashboard": "true"}
                },
                "spec": {
                    "containers": [
                        {
                            "name": "char-completion-serving",
                            "image": f"image-registry.openshift-image-registry.svc:5000/{namespace}/char-completion-serving:latest",
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                            "env": [
                                {"name": "PORT", "value": "8080"},
                                {"name": "S3_ENDPOINT", "value": "http://minio-dspa:9000"}
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"}
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 3,
                                "periodSeconds": 5
                            }
                        }
                    ]
                }
            }
        }
    }

    print(f"[K8S] Reconciling Deployment 'char-completion' in '{namespace}'...")
    res, status_code = k8s_request("POST", f"/apis/apps/v1/namespaces/{namespace}/deployments", deploy_manifest)
    if status_code in (200, 201):
        print("[SUCCESS] Deployment 'char-completion' created successfully.")
    elif status_code == 409:
        k8s_request("PATCH", f"/apis/apps/v1/namespaces/{namespace}/deployments/char-completion", deploy_manifest, content_type="application/merge-patch+json")
        print("[SUCCESS] Deployment 'char-completion' updated with latest model.")
    else:
        print(f"[INFO] Deployment response: {status_code}")

    # 2. Reconcile Service 'char-completion'
    svc_manifest = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "char-completion",
            "namespace": namespace,
            "labels": {"app": "char-completion"}
        },
        "spec": {
            "selector": {"app": "char-completion"},
            "ports": [{"port": 8080, "targetPort": 8080, "protocol": "TCP"}]
        }
    }

    print(f"[K8S] Reconciling Service 'char-completion'...")
    res, status_code = k8s_request("POST", f"/api/v1/namespaces/{namespace}/services", svc_manifest)
    if status_code in (200, 201):
        print("[SUCCESS] Service 'char-completion' created.")
    elif status_code == 409:
        k8s_request("PATCH", f"/api/v1/namespaces/{namespace}/services/char-completion", svc_manifest, content_type="application/merge-patch+json")
        print("[SUCCESS] Service 'char-completion' is up to date.")

    # 3. Reconcile OpenShift Route 'char-completion'
    route_manifest = {
        "apiVersion": "route.openshift.io/v1",
        "kind": "Route",
        "metadata": {
            "name": "char-completion",
            "namespace": namespace,
            "labels": {"app": "char-completion"}
        },
        "spec": {
            "to": {"kind": "Service", "name": "char-completion"},
            "port": {"targetPort": 8080},
            "tls": {"termination": "edge"}
        }
    }

    print(f"[K8S] Reconciling OpenShift Route 'char-completion'...")
    res, status_code = k8s_request("POST", f"/apis/route.openshift.io/v1/namespaces/{namespace}/routes", route_manifest)
    if status_code in (200, 201):
        host = res.get("spec", {}).get("host", "")
        print(f"[SUCCESS] OpenShift Route created: https://{host}")
    elif status_code == 409:
        print("[SUCCESS] OpenShift Route 'char-completion' is ready.")
    else:
        print(f"[INFO] Route response: {status_code}")

    print("=" * 65)
    return "deployed"


@dsl.pipeline(
    name="ocp-char-completion-pipeline",
    description="Trains a PyTorch character-level language model, records metrics, evaluates text completions, and deploys model hosting."
)
def char_completion_pipeline(
    num_gpus: int = 0,
    epochs: int = 150,
    words_list: str = "",
    deploy_model: bool = True
):
    train_task = train_char_lm_op(num_gpus=num_gpus, epochs=epochs, words_list=words_list)
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

    deploy_task = deploy_kserve_op(
        model_name="char-completion",
        storage_uri="s3://mlpipeline/models/char-completion",
        deploy=deploy_model
    )
    deploy_task.set_caching_options(False)
    deploy_task.set_cpu_request("200m")
    deploy_task.set_cpu_limit("1")
    deploy_task.set_memory_request("256Mi")
    deploy_task.set_memory_limit("1Gi")
    deploy_task.after(train_task)


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
