# AI Model Usage & Inference Guide

This guide explains how the **Character-Level Language Model (CharLSTM)** works, how the pipeline trains and exports it, and how to use it locally in Python with [`infer.py`](file://infer.py).

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

## 2. Pipeline Execution in OpenShift AI

When the Kubeflow pipeline runs, it executes two connected steps:

```
┌─────────────────────────────────┐
│ Step 1: train-char-lm-op        │
│                                 │
│  - Trains CharLSTM on vocabulary│
│  - Emits scalar metrics:        │
│      * final_loss               │
│      * epochs                   │
│      * vocab_size               │
│      * train_time_ms            │
│  - Exports model.pt & vocab.json│
└────────────────┬────────────────┘
                 │
                 │ Passes model_artifact (S3 / MinIO)
                 ▼
┌─────────────────────────────────┐
│ Step 2: evaluate-completions-op │
│                                 │
│  - Ingests model_artifact       │
│  - Evaluates sample prefixes:   │
│      * 'ope'  --> 'openshift'   │
│      * 'kube' --> 'kubeflow'    │
│      * 'py'   --> 'pytorch'     │
│      * 'con'  --> 'container'   │
│  - Writes evaluation report     │
└─────────────────────────────────┘
```

### Viewing in the OpenShift AI Console:
* **Metrics Tab:** Click the **Metrics** tab on your pipeline run to see the training loss curve and scalar summary metrics.
* **Artifacts Tab:** Click on **Artifacts** to see `model_artifact` stored in S3/MinIO, complete with framework metadata (`PyTorch`, `CharLSTM`).

---

## 3. Local Inference with `infer.py`

You can interact with the model directly from your local terminal using [`infer.py`](file://infer.py).

> 💡 **Automatic Cluster Download:** If `model.pt` is not found locally, `infer.py` will automatically connect to your OpenShift cluster and download the latest trained model artifact from your pipeline run! You can also force a re-download anytime using `--from-cluster`.

### Prerequisites
Make sure your virtual environment is active:
```bash
source .venv/bin/activate
pip install torch
```

### Single Word Completion
```bash
python infer.py --prompt "ope"
```
Output:
```text
=============================================
Input Prefix  : 'ope'
Predicted Word: 'openshift'
=============================================
```

```bash
python infer.py --prompt "kube"
# Predicted Word: 'kubeflow'

python infer.py --prompt "py"
# Predicted Word: 'pytorch'
```

### Interactive Autocomplete Mode
Launch a real-time prompt loop where you can test different prefixes interactively:
```bash
python infer.py --interactive
```
Output:
```text
============================================================
🤖 Interactive AI Word Completion
Type a prefix (e.g. 'ope', 'kube', 'py', 'ten') and press Enter.
Type 'exit' or press Ctrl+C to quit.
============================================================

Prompt > ope
  └── Predicted completion: 'openshift'

Prompt > kube
  └── Predicted completion: 'kubeflow'

Prompt > con
  └── Predicted completion: 'container'

Prompt > exit
Goodbye!
```

---

## 4. Customizing the Training Vocabulary

To train the model on your own company or project terms:
1. Open [`src/model.py`](file://src/model.py).
2. Modify `DEFAULT_VOCAB_WORDS`:
   ```python
   DEFAULT_VOCAB_WORDS = [
       "openshift",
       "kubernetes",
       "yourappname",
       "datacenter",
       "microservice",
   ]
   ```
3. Re-run `python pipeline/submit_pipeline.py` to trigger a new pipeline run with your updated vocabulary!
