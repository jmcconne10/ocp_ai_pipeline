# OpenShift AI Kubeflow Pipeline Template

A production-ready, reusable template for developing, compiling, and running **Kubeflow Pipelines (KFP v2)** on **Red Hat OpenShift AI (RHOAI)** directly from local IDEs and native OpenShift CI/CD (Tekton).

> ⚡ **Looking to get up and running immediately?**  
> Jump straight to the **[Quick Setup Guide (QUICKSTART.md)](QUICKSTART.md)** to configure your OpenShift environment and run your first pipeline in under 2 minutes with `python3 setup.py`.

---

## Architecture & Features

```
[ Local IDE / Developer CLI ]                     [ OpenShift Pipelines (Tekton) ]
       │                                                         │
       │  (Bearer Token / oc whoami -t)                          │  (In-Cluster ServiceAccount Token)
       ▼                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ OpenShift AI (RHOAI) Data Science Project Namespace                              │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ Data Science Pipelines Application (DSPA 2.x)                            │   │
│   │  ├── DSPA API Server Route (OAuth Proxy)                                 │   │
│   │  ├── Workflow Controller & Persistence Agent                             │   │
│   │  └── MariaDB Metadata Store                                              │   │
│   └───────────────────────────────┬──────────────────────────────────────────┘   │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ Object Storage (Pipeline Artifacts & Logs)                               │   │
│   │  ├── Trial / Self-Contained : In-Cluster MinIO                           │   │
│   │  └── Company / Enterprise    : AWS S3 / Ceph / OpenShift Data Foundation  │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ Workload Pod Execution                                                   │   │
│   │  └── GPU Task: `nvidia.com/gpu: 1` (or CPU fallback: `num_gpus: 0`)      │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Target Platform:** OpenShift AI (Data Science Pipelines 2.x / KFP v2 SDK).
- **GPU Workload:** PyTorch CUDA device discovery, memory introspection, and matrix multiplication benchmark.
- **Trial / 0-GPU Fallback:** Cleanly supports running on CPU-only trial clusters with `--num-gpus 0`.
- **Zero External Secrets in CI/CD:** Supports 100% native **OpenShift Pipelines (Tekton)** using in-cluster `ServiceAccount` authentication.
- **Dual CI/CD Support:** Includes OpenShift Tekton manifests as well as a GitHub Actions workflow.

---

## Directory Structure

```
├── .env.example                     # Environment template for local CLI runs
├── .gitignore                       # Git ignore rules for Python, KFP artifacts, and secrets
├── Dockerfile                       # Container image for src/main.py workload
├── manifests/
│   ├── dspa.yaml                    # Data Science Pipelines Application Custom Resource
│   ├── minio.yaml                   # In-cluster MinIO storage manifest
│   └── networkpolicy-minio.yaml     # NetworkPolicy for operator health-checks in hardened environments
├── pipeline/
│   ├── pipeline.py                  # KFP v2 pipeline definition & compiler
│   ├── requirements.txt             # Local/CI runner dependencies (kfp, python-dotenv)
│   └── submit_pipeline.py           # CLI submission script with auto token discovery
├── src/
│   ├── main.py                      # Standalone PyTorch GPU discovery & computation workload
│   └── requirements.txt             # Container runtime workload dependencies
└── tekton/
    ├── rbac.yaml                    # ServiceAccount & RoleBinding for in-cluster execution
    ├── task-compile-submit.yaml     # Tekton Task for KFP compilation & submission
    ├── pipeline.yaml                # Tekton Pipeline linking git clone and submission
    └── pipelinerun.yaml             # Example execution trigger
```

---

## Quick Start (Automated Setup)

To automatically configure your OpenShift cluster, deploy MinIO storage, provision the DSPA pipeline server, generate your `.env`, and compile the pipeline in one interactive step, run:

```bash
python3 setup.py
```

The script will prompt for:
1. **OpenShift Token** (or reuse an active `oc` session)
2. **Project / Namespace**
3. **Storage preference** (In-Cluster MinIO vs External S3)
4. Automatic `.env` generation and initial pipeline compilation!

To completely clean up and reset your cluster and local state at any time:
```bash
python3 destroy.py
```

---

## Manual Cluster Setup & Reproducibility Guide

If you prefer to configure resources manually, follow the steps below:

### 1.1 Verify OpenShift AI Operator
Ensure **Red Hat OpenShift AI** is installed by your cluster administrator. Verify in your terminal:
```bash
oc api-resources | grep datasciencepipelinesapplications
```

### 1.2 Object Storage Setup

Kubeflow Pipelines requires an S3-compatible bucket for artifacts. Choose the option matching your environment:

#### Option A: In-Cluster MinIO (Self-Contained / Trial Environments)
If your company or trial cluster does not provide an external AWS S3 bucket, apply the included MinIO deployment and NetworkPolicy:
```bash
# 1. Apply MinIO storage and credentials secret
oc apply -f manifests/minio.yaml

# 2. (Optional / Hardened clusters) Apply NetworkPolicy allowing operator health-checks
oc apply -f manifests/networkpolicy-minio.yaml

# 3. Apply the Data Science Pipelines Application (DSPA)
oc apply -f manifests/dspa.yaml
```

#### Option B: Company S3 / Ceph / AWS S3 (Enterprise Production)
If your company provides an S3 bucket:
1. Create the credentials secret in your OpenShift namespace:
   ```bash
   oc create secret generic pipeline-s3-secret \
     --from-literal=accesskey="<YOUR_AWS_ACCESS_KEY>" \
     --from-literal=secretkey="<YOUR_AWS_SECRET_KEY>"
   ```
2. Update `manifests/dspa.yaml` under `spec.objectStorage.externalStorage`:
   ```yaml
   spec:
     objectStorage:
       externalStorage:
         host: s3.amazonaws.com          # Or company Ceph/MinIO endpoint
         bucket: your-company-bucket
         region: us-east-1
         scheme: https
         secure: true
         s3CredentialsSecret:
           secretName: pipeline-s3-secret
           accessKey: accesskey
           secretKey: secretkey
   ```
3. Apply the DSPA:
   ```bash
   oc apply -f manifests/dspa.yaml
   ```

### 1.3 Verify DSPA is Ready
Check that all DSPA pods are running:
```bash
oc get pods -l app.kubernetes.io/name=data-science-pipelines
```
Retrieve your DSPA route URL:
```bash
oc get route -l app.kubernetes.io/name=data-science-pipelines -o jsonpath='{"https://"}{.items[0].spec.host}{"\n"}'
```

---

## Step 2: Local Developer Setup

### 2.1 Set Up Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r pipeline/requirements.txt
```

### 2.2 Configure Local Environment (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in the values in `.env`:
```env
# Extract from terminal: oc whoami -t
OCP_TOKEN=sha256~...

# Extract from terminal: oc get route -l app.kubernetes.io/name=data-science-pipelines -o jsonpath='{"https://"}{.items[0].spec.host}'
DSPA_HOST=https://ds-pipeline-dspa-your-namespace.apps.yourcluster.com

# Your project namespace
NAMESPACE=your-namespace

# Set to False if using trial / self-signed cluster certificates
VERIFY_SSL=True

# Set to 0 if your cluster lacks physical GPUs, or 1 on GPU nodes
NUM_GPUS=1
```

---

## Step 3: Compiling & Submitting Pipelines Locally

### 3.1 Compile Pipeline to YAML
To inspect or compile the pipeline definition locally:
```bash
python pipeline/pipeline.py --output pipeline.yaml
```

### 3.2 Submit Pipeline to OpenShift AI
Submit a run directly against your OpenShift cluster using `submit_pipeline.py`:

```bash
# Standard run (uses settings from .env)
python pipeline/submit_pipeline.py

# Run on trial / CPU-only cluster (explicit 0 GPUs)
python pipeline/submit_pipeline.py --num-gpus 0

# Run with self-signed certificate bypass
python pipeline/submit_pipeline.py --insecure

# Custom experiment and run name
python pipeline/submit_pipeline.py \
  --experiment-name "model-training-v1" \
  --run-name "gpu-experiment-trial-1" \
  --num-gpus 1
```

Upon submission, the CLI will output:
```
🚀 Pipeline submitted successfully!
Run ID        : 7f81b37e-8c34-42b7-a3f2-1d599b1a0e1c
Experiment ID : 2d62884f-4d92-4889-b51c-74f00b411d95
Status Details: https://ds-pipeline-dspa.../#/runs/details/7f81b37e-8c34-42b7-a3f2-1d599b1a0e1c
```

---

## Step 4: OpenShift Native CI/CD (Tekton Pipelines)

If your company does not use GitHub Actions, you can execute the entire pipeline lifecycle natively inside OpenShift using **Red Hat OpenShift Pipelines (Tekton)**.

### 4.1 Install OpenShift Pipelines Operator
Ensure the **Red Hat OpenShift Pipelines Operator** is installed via **OperatorHub** in the OpenShift Web Console.

### 4.2 Apply Tekton Resources
```bash
# 1. Create ServiceAccount & RBAC permissions
oc apply -f tekton/rbac.yaml

# 2. Create the compilation & submission task
oc apply -f tekton/task-compile-submit.yaml

# 3. Create the pipeline
oc apply -f tekton/pipeline.yaml
```

### 4.3 Trigger an In-Cluster Pipeline Run
```bash
oc create -f tekton/pipelinerun.yaml
```
Or monitor using the `tkn` CLI:
```bash
tkn pipelinerun logs -f -L
```

---

## Step 5: Useful OpenShift Diagnostic Commands

### Check DSPA Route
```bash
oc get route -l app.kubernetes.io/name=data-science-pipelines -o jsonpath='{"https://"}{.items[0].spec.host}{"\n"}'
```

### Extract Your Current Login Token
```bash
oc whoami -t
```

### Check GPU Nodes & Drivers
```bash
# Check if NVIDIA GPU nodes exist on the cluster
oc get nodes -l nvidia.com/gpu.present=true

# Check allocatable GPU count across all nodes
oc describe nodes | grep -E "nvidia.com/gpu:|Name:"

# Check NVIDIA GPU Operator pods
oc get pods -n nvidia-gpu-operator
```

### Check Namespace Resource Quotas (GPU Limits)
```bash
oc get resourcequota -o yaml | grep -i gpu
```
If `limits.nvidia.com/gpu: "0"`, the namespace is restricted to CPU workloads—ensure you pass `--num-gpus 0` when submitting.

---

## Step 6: GitHub Actions CI/CD (Alternative)

For environments with GitHub Actions enabled:
1. Navigate to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**.
2. Add the following repository secrets:
   - `OPENSHIFT_SERVER`: The API URL of your OpenShift cluster (e.g. `https://api.mycluster.domain:6443`).
   - `OPENSHIFT_TOKEN`: ServiceAccount or user token.
   - `DSPA_ROUTE_URL`: Full HTTPS URL to your DSPA route.
3. Every push or pull request to `main` will automatically compile and submit the pipeline.
