# AI Model Usage & Inference Guide

This guide explains how the **Character-Level Language Model (CharLSTM)** works, how the 3-step pipeline trains, exports, and hosts it, and how to query it via the local CLI ([`infer.py`](file://infer.py)) or the hosted API client ([`client/test_api.py`](file://client/test_api.py)).

---

## 1. How the Model Works

The model is a **Character-Level LSTM Language Model** implemented in PyTorch:
- **Embedding Layer:** Converts character tokens into continuous vector representations.
- **LSTM Layer:** Maintains a recurrent hidden state representing the sequence of characters typed so far.
- **Linear Projection:** Predicts the probability distribution over the vocabulary for the *next* character.

When you supply a prefix like `"ope"`:
1. The model ingests `'o'`, `'p'`, `'e'` to update its internal hidden state.
2. It iteratively predicts the most probable next character (`'n'`, `'s'`, `'h'`, `'i'`, `'f'`, `'t'`) until it reaches the end-of-word token (`'$'`).
3. Output: `"openshift"`.

---

## 2. 3-Step Pipeline in OpenShift AI

When the Kubeflow pipeline runs, it executes three connected steps:

```
┌─────────────────────────────────┐
│ Step 1: train-char-lm-op        │
│                                 │
│  - Trains CharLSTM on vocabulary│
│  - Emits scalar metrics         │
│  - Exports to S3 model storage  │
└────────────────┬────────────────┘
                 │
                 │ Passes model_artifact
                 ▼
┌─────────────────────────────────┐
│ Step 2: evaluate-completions-op │
│                                 │
│  - Ingests model_artifact       │
│  - Evaluates test prefixes      │
│  - Writes evaluation report     │
└────────────────┬────────────────┘
                 │
                 │ Trigger after training
                 ▼
┌─────────────────────────────────┐
│ Step 3: deploy-kserve-op        │
│                                 │
│  - Reconciles KServe Secret     │
│  - Reconciles ServingRuntime    │
│  - Reconciles InferenceService  │
│  - Points to fixed S3 path      │
└─────────────────────────────────┘
```

---

## 3. Querying the Hosted Model API (`client/test_api.py`)

Once deployed by the pipeline, test the endpoint with the client tool:

```bash
# Auto-discovers the OpenShift route:
python client/test_api.py --prefix "ope"

# Enterprise KServe v2 protocol:
python client/test_api.py --kserve --prefix "kube"

# Interactive CLI prompt:
python client/test_api.py
```

Or query via curl:
```bash
curl -X POST https://<ROUTE_HOST>/complete \
  -H "Content-Type: application/json" \
  -d '{"prefix": "con"}'
```

---

## 4. Local Inference with `infer.py`

You can also run local completions without sending requests to the cluster:

```bash
python infer.py --prompt "ope"
python infer.py --interactive
```

If local model weights are missing, `infer.py` automatically downloads the latest checkpoint from your OpenShift pipeline MinIO storage!

---

## 5. Customizing Training Words

Customize vocabulary using either:
1. **File:** Edit [`data/words.txt`](file://data/words.txt).
2. **CLI argument:**
   ```bash
   python pipeline/submit_pipeline.py --words "cloud,openshift,microservice,container"
   ```
3. **Local training script:**
   ```bash
   python training/train.py --words-file data/words.txt
   ```
