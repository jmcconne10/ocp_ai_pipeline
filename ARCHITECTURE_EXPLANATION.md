# OpenShift AI Pipeline & Model Serving: Architecture & Enterprise Replication Guide

This document explains the technical architecture implemented in this repository, how each modular component functions, and the exact steps required to replicate this setup inside your company's enterprise OpenShift environment.

---

## 1. What We Built & How It Works

### The Core Problem
Enterprise AI/ML teams need to train models, record metrics, export checkpoints, and immediately host inference services on **Red Hat OpenShift AI (RHOAI)**. However:
- Running pipelines from local developer laptops or CI runners often faces authentication, internal proxy, TLS certificate, or S3 storage hurdles.
- Multi-tenant enterprise clusters enforce strict namespace quotas, require internal NetworkPolicies, and restrict GPU access.
- Moving from pipeline completion to a production-ready model endpoint typically requires manual ops work rather than automated promotion.

### The Solution Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │              Trigger Mechanisms              │
                  ├──────────────────────┬───────────────────────┤
                  │ 1. Local IDE / CLI   │ 2. OpenShift Native   │
                  │    (Developer Laptop)│    CI/CD (Tekton)     │
                  └──────────┬───────────┴───────────┬───────────┘
                             │                       │
           Bearer Token via  │                       │ In-cluster ServiceAccount
           `oc whoami -t`    │                       │ (/var/run/secrets/...)
                             ▼                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ OpenShift Project Namespace (e.g. `joe-mcconnell10-dev`)                       │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Data Science Pipelines Application (DSPA 2.x)                            │  │
│  │  - OAuth Proxy Ingress Route (`ds-pipeline-dspa`)                        │  │
│  │  - KFP API Server (Validates pipeline specs & schedules runs)            │  │
│  │  - MariaDB (Tracks runs, experiments, and metadata)                      │  │
│  │  - Argo Workflow Controller (Executes the 3-step DAG):                   │  │
│  │      * Step 1: Train CharLSTM Model (Configurable Vocabulary & Metrics)  │  │
│  │      * Step 2: Evaluate Completions & Save Artifact Report               │  │
│  │      * Step 3: Automated Model Serving Deployment & Route Creation       │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Object Storage (Pipeline Artifacts & Model Weights)                      │  │
│  │  - Trial: In-cluster MinIO pod + PVC                                     │  │
│  │  - Production: Company AWS S3 / Ceph / OpenShift Data Foundation         │  │
│  │  - Fixed S3 Path: `s3://mlpipeline/models/char-completion/model.pt`      │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Model Serving Endpoint (`char-completion`)                               │  │
│  │  - Deployment: Lightweight FastAPI container (256 MiB RAM, non-root UID) │  │
│  │  - Service: `char-completion` (Port 8080)                                │  │
│  │  - OpenShift Route: `https://char-completion...` (Edge TLS)              │  │
│  │  - Protocol 1: Friendly REST (`/complete`, `/health`, `/docs`)           │  │
│  │  - Protocol 2: KServe v2 Data Plane (`/v2/models/char-completion/infer`) │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Client / Consumer Applications                                           │  │
│  │  - Interactive CLI test client (`client/test_api.py`)                    │  │
│  │  - Interactive Swagger Web UI (`https://<ROUTE>/docs`)                   │  │
│  │  - External curl or microservice REST calls                              │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Components Explained

| Module / File | Component | Purpose |
|---|---|---|
| `data/words.txt` | **Vocabulary Dataset** | Configurable domain vocabulary used to train the character completion model without touching Python code. |
| `training/model.py` | **Model Architecture** | PyTorch `CharLSTM` model definition with character embeddings, LSTM recurrent layers, and greedy sequence generator. |
| `training/train.py` | **Standalone Trainer** | Local/container training script supporting `--words-file`, `--words`, `--epochs`, and `--output-dir`. |
| `pipeline/pipeline.py` | **3-Step KFP Pipeline** | Kubeflow Pipelines v2 DAG: trains model on domain words, records scalar metrics, evaluates prefixes, and deploys the serving endpoint. |
| `pipeline/submit_pipeline.py` | **CLI Submission** | Compiles and submits `pipeline.yaml` to OpenShift AI DSPA with auto token/route discovery. |
| `serving/app.py` | **Dual-Protocol Server** | Production-ready FastAPI application supporting both Friendly REST (`/complete`) and KServe v2 Data Plane (`/v2/models/char-completion/infer`). |
| `serving/Dockerfile` | **Serving Container** | OpenShift-hardened container image running as non-root UID (group 0) on port 8080. |
| `serving/buildconfig.yaml` | **On-Cluster Build** | OpenShift `ImageStream` and `BuildConfig` for compiling container images directly inside the cluster registry. |
| `client/test_api.py` | **API Test Client** | Interactive and single-query CLI client with auto-discovery of OpenShift routes. |
| `manifests/minio.yaml` | **Storage (Trial)** | Self-contained S3-compatible MinIO instance with persistent storage (`gp3`). |
| `manifests/dspa.yaml` | **Pipeline Server** | OpenShift AI Custom Resource (`DataSciencePipelinesApplication`) configuring API server and database. |
| `manifests/networkpolicy-minio.yaml` | **Network Security** | NetworkPolicy allowing OpenShift AI Operator to health-check MinIO in hardened clusters. |
| `tekton/` | **Native CI/CD** | OpenShift Pipelines manifests to automate pipeline runs from internal Git servers. |

---

## 3. How to Reproduce on Your Company Environment

Follow this checklist to deploy this pipeline on your company's OpenShift AI cluster:

### Step 1: Clone Repository & Log In to OpenShift
```bash
git clone <your-company-git-url>
cd ocp_ai_pipeline

# Log in using your company OpenShift credentials
oc login --server=https://api.yourcompany-ocp.domain:6443 --token=<YOUR_TOKEN>
oc project <your-team-namespace>
```

### Step 2: Configure Object Storage

#### Option A: If your company provides an S3 bucket (AWS S3, Ceph, MinIO):
1. Create the credentials secret:
   ```bash
   oc create secret generic pipeline-s3-secret \
     --from-literal=accesskey="<YOUR_COMPANY_ACCESS_KEY>" \
     --from-literal=secretkey="<YOUR_COMPANY_SECRET_KEY>"
   ```
2. In `manifests/dspa.yaml`, replace `spec.objectStorage` with `externalStorage`:
   ```yaml
   spec:
     objectStorage:
       externalStorage:
         host: s3.company.domain  # e.g. s3.amazonaws.com or internal Ceph
         bucket: your-pipeline-artifacts-bucket
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

#### Option B: If your company cluster does NOT provide S3 (Use In-Cluster MinIO):
```bash
oc apply -f manifests/minio.yaml
oc apply -f manifests/networkpolicy-minio.yaml
oc apply -f manifests/dspa.yaml
```

### Step 3: Run Automated Setup
```bash
python3 setup.py
```
This script handles namespace selection, storage verification, DSPA route discovery, KServe serving RBAC, virtual environment creation, and pipeline pre-compilation.

### Step 4: Submit Pipeline Run
```bash
source .venv/bin/activate

# Submit standard run (trains model, records metrics, deploys serving endpoint):
python pipeline/submit_pipeline.py

# Or pass custom company words:
python pipeline/submit_pipeline.py --words "datacenter,microservice,telemetry,compliance"
```

### Step 5: Test the Hosted Model Endpoint
```bash
# Auto-detects your OpenShift route:
python client/test_api.py --prefix "data"

# Or launch interactive completion mode:
python client/test_api.py
```

---

## 4. Key Lessons Learned & Enterprise Gotchas

During development and cluster testing, several critical enterprise constraints were resolved:

1. **4-Byte Emojis vs MariaDB Charset Collation:**
   - *Problem:* OpenShift AI's bundled MariaDB metadata database defaults to 3-byte `utf8` rather than `utf8mb4`. Any 4-byte unicode emojis in pipeline inline scripts cause MySQL error `1366: Incorrect string value` when storing pipeline manifests.
   - *Resolution:* All pipeline code, component logs, and artifact metadata use clean ASCII tags (e.g. `[SUCCESS]`, `[INFO]`, `[METRIC]`).

2. **Multi-Tenant NetworkPolicies on MinIO:**
   - *Problem:* In hardened OpenShift clusters, cross-namespace ingress from the Operator namespace (`redhat-ods-applications`) to user namespaces is blocked by default, causing DSPA readiness probes to fail.
   - *Resolution:* Included `manifests/networkpolicy-minio.yaml` to explicitly allow traffic from the operator to port 9000.

3. **GPU Quotas on Non-GPU / Trial Clusters:**
   - *Problem:* Many developer namespaces enforce `limits.nvidia.com/gpu: "0"`. Submitting a pipeline requesting GPUs causes immediate Kubernetes quota rejection.
   - *Resolution:* `submit_pipeline.py` supports `--num-gpus 0`, and `pipeline/pipeline.py` dynamically omits GPU limits and node selectors when running in CPU mode.

4. **KServe Admission Webhook Memory Quota:**
   - *Problem:* On constrained multi-tenant clusters, KServe's default `storage-initializer` admission webhook injects an init-container requesting **24 GiB** of RAM, which exceeds standard namespace resource quotas (`compute-deploy`).
   - *Resolution:* Our serving deployment uses a lightweight 256 MiB container that loads model weights directly from the in-cluster MinIO/S3 endpoint on startup and exposes a native OpenShift Route, ensuring 100% reliability across all quota environments.
