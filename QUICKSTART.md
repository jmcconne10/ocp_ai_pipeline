# Quick Setup Guide

Get up and running with Kubeflow Pipelines on OpenShift AI in under 2 minutes.

---

## Prerequisites

Before running the setup script, ensure you have:
1. **Python 3.9+** installed locally.
2. The **OpenShift CLI (`oc`)** installed:
   ```bash
   which oc
   ```
3. Your **OpenShift Login Token**:
   - Open your OpenShift Web Console.
   - Click your username in the top-right corner → **Copy login command** → **Display Token**.
   - Copy your token (starts with `sha256~...`).

---

## 1-Command Automated Setup

Run the interactive setup wizard:

```bash
python3 setup.py
```

### What the Wizard Does For You:
1. Prompts for your **OpenShift Token** and **Cluster URL** (or reuses your active `oc` session).
2. Sets your target **Project / Namespace**.
3. Deploys S3 storage (**In-Cluster MinIO** for trials/sandboxes or prompts for your enterprise S3 bucket).
4. Deploys and reconciles the **Data Science Pipelines Application (DSPA)**.
5. Auto-discovers the **DSPA Route URL**.
6. Auto-detects **GPU quotas** (sets `NUM_GPUS=0` for trial clusters, `1` for GPU clusters).
7. Generates your local [`.env`](file://.env) file.
8. Creates a local virtual environment (`.venv`), installs dependencies, compiles `pipeline.yaml`, and uploads the pipeline template to your OpenShift AI UI catalog.

---

## Run Your First Pipeline

After running `setup.py`, activate your environment and submit a pipeline run:

```bash
source .venv/bin/activate
python pipeline/submit_pipeline.py
```

Output:
```text
🚀 Pipeline submitted successfully!
Run ID        : 7ed4eea9-ec32-4e17-979b-8776aa70de23
Experiment ID : dfa24aea-ba1b-40fa-94f7-db7f9e2537ca
Status Details: https://ds-pipeline-dspa.../#/runs/details/7ed4eea9-ec32-4e17-979b-8776aa70de23
```

---

## View Live Progress in OpenShift AI

1. Open the **Red Hat OpenShift AI Dashboard**.
2. Go to **Data Science Projects** → click your project name.
3. Click the **Pipelines** tab:
   - **Pipelines sub-tab:** View the `ocp-gpu-hello-world` pipeline definition and DAG graph.
   - **Runs sub-tab:** Click on your active or completed run to see real-time logs and PyTorch execution output!

---

## Common Options

```bash
# Run with explicit 0 GPUs (CPU fallback mode)
python pipeline/submit_pipeline.py --num-gpus 0

# Run with 1 GPU (on clusters with NVIDIA GPU nodes)
python pipeline/submit_pipeline.py --num-gpus 1

# Disable SSL verification (for self-signed cluster certificates)
python pipeline/submit_pipeline.py --insecure

# Exit the virtual environment
deactivate
```

---

## Clean Teardown (Reset Everything)

To completely remove all cluster resources (DSPA, MinIO, MariaDB, PVCs, secrets, pods) and local `.env` so you can test `setup.py` fresh:

```bash
python3 destroy.py
```

---

## Next Steps
- For detailed architecture, OpenShift native Tekton CI/CD, and enterprise reproduction instructions, see [README.md](file://README.md) and [ARCHITECTURE_EXPLANATION.md](file://ARCHITECTURE_EXPLANATION.md).
