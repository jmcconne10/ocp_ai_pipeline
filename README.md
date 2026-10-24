# OpenShift AI Kubeflow Pipeline & Model Serving Template

A production-ready, modular template for developing, compiling, and running **Kubeflow Pipelines (KFP v2)** and hosting models with **KServe** on **Red Hat OpenShift AI (RHOAI)**.

> ⚡ **Looking to get up and running immediately?**  
> Run `python3 setup.py` to configure your OpenShift namespace, storage, KServe serving, and compile the pipeline in under 2 minutes. See **[Quick Setup Guide (QUICKSTART.md)](QUICKSTART.md)**.

---

## 🏗️ Architecture & Features

```
[ Local IDE / Developer CLI ]                     [ OpenShift Pipelines (Tekton) ]
       │                                                         │
       │  (Bearer Token / oc whoami -t)                          │  (In-Cluster ServiceAccount Token)
       ▼                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ OpenShift AI (RHOAI) Data Science Project Namespace                              │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ Data Science Pipelines Application (DSPA 2.x)                              │  │
│  │  ├── Step 1: Train CharLSTM Model (Configurable Vocabulary & Metrics)      │  │
│  │  ├── Step 2: Evaluate Completions & Save Artifact Report                   │  │
│  │  └── Step 3: Automated KServe Deployment (Reconciles Model Endpoint)       │  │
│  └─────────────────────────────────┬──────────────────────────────────────────┘  │
│                                    │                                             │
│                                    ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ Object Storage (Pipeline Artifacts & Model Checkpoints)                    │  │
│  │  ├── S3 Model Checkpoint: `s3://mlpipeline/models/char-completion/model.pt`│  │
│  │  └── S3 Vocab Map       : `s3://mlpipeline/models/char-completion/vocab...`│  │
│  └─────────────────────────────────┬──────────────────────────────────────────┘  │
│                                    │                                             │
│                                    ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ KServe v1beta1 Model Hosting (RawDeployment Mode)                          │  │
│  │  ├── InferenceService: `char-completion`                                   │  │
│  │  ├── Protocol 1: Friendly REST (`/complete`, `/health`, `/docs`)           │  │
│  │  └── Protocol 2: KServe v2 Data Plane (`/v2/models/char-completion/infer`) │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Target Platform:** OpenShift AI (Data Science Pipelines 2.x / KFP v2 SDK & KServe v1beta1).
- **Modular Directory Layout:** Decoupled modules for training, pipelines, serving, client API testing, and infrastructure.
- **Customizable Vocabulary:** Train on domain-specific vocabulary via `data/words.txt`, CLI flags, or pipeline parameters.
- **Automated KServe Deployment:** Pipeline step 3 automatically reconciles KServe `InferenceService`, `ServingRuntime`, and S3 Secrets.
- **Dual API Support:** Serving application supports both standard KServe v2 Data Plane and developer-friendly REST endpoints.
- **Enterprise Network Ready:** Configurable base image mirrors, SSL certificate handling (`VERIFY_SSL`), and sanitized ASCII logs preventing database collation errors.

---

## 📁 Repository Structure

```
├── client/                          # Standalone API test client & verification scripts
│   ├── README.md                    # Curl and Python API usage examples
│   └── test_api.py                  # CLI test client (REST & KServe v2 protocols)
├── data/
│   └── words.txt                    # Customizable domain vocabulary dataset
├── manifests/
│   ├── dspa.yaml                    # Data Science Pipelines Application Custom Resource
│   ├── minio.yaml                   # In-cluster MinIO storage manifest
│   └── networkpolicy-minio.yaml     # NetworkPolicy for operator health-checks in hardened clusters
├── pipeline/
│   ├── pipeline.py                  # 3-step KFP v2 pipeline definition & compiler
│   ├── requirements.txt             # Local/CI runner dependencies (kfp, python-dotenv)
│   └── submit_pipeline.py           # CLI submission script with custom words and deploy flags
├── serving/
│   ├── Dockerfile                   # Serving container image definition
│   ├── app.py                       # FastAPI serving application (REST & KServe v2)
│   ├── buildconfig.yaml             # OpenShift BuildConfig & ImageStream for on-cluster builds
│   ├── inferenceservice.yaml        # KServe InferenceService manifest (RawDeployment)
│   ├── kserve-secret.yaml           # S3 DataConnection Secret & ServiceAccount template
│   ├── requirements.txt             # Serving dependencies (fastapi, uvicorn, pydantic, torch)
│   └── servingruntime.yaml          # KServe ServingRuntime manifest
├── tekton/                          # OpenShift Pipelines native CI/CD manifests
│   ├── rbac.yaml                    # ServiceAccount & RoleBinding for in-cluster execution
│   ├── task-compile-submit.yaml     # Tekton Task for KFP compilation & submission
│   ├── pipeline.yaml                # Tekton Pipeline linking git clone and submission
│   └── pipelinerun.yaml             # Example execution trigger
├── training/
│   ├── Dockerfile                   # Standalone training container image
│   ├── model.py                     # PyTorch CharLSTM architecture & completion generator
│   ├── requirements.txt             # Training dependencies (torch)
│   └── train.py                     # Standalone CLI training script
├── destroy.py                       # Automated teardown script for all cluster & local resources
├── infer.py                         # Local / cluster model download and word completion CLI
└── setup.py                         # Interactive setup wizard for OpenShift & local development
```

---

## ⚡ Quick Start

### 1. Run Automated Setup
```bash
python3 setup.py
```
This script will:
- Authenticate to your OpenShift cluster
- Deploy MinIO object storage and the DSPA pipeline server
- Configure KServe RBAC, ServingRuntime, and ImageStream
- Generate `.env` and compile `pipeline.yaml`

### 2. Submit the Pipeline
```bash
source .venv/bin/activate
python pipeline/submit_pipeline.py
```

### 3. Test the Hosted Model API
```bash
# Interactive completion prompt (auto-discovers OpenShift route)
python client/test_api.py

# Single query via friendly REST API
python client/test_api.py --prefix "ope"

# Query via KServe v2 Data Plane standard
python client/test_api.py --kserve --prefix "kube"
```

---

## 🎯 Customizable Training Vocabulary

You can customize the vocabulary words used to train the character completion model without modifying Python code:

### Option A: Edit the Dataset File
Modify `data/words.txt` directly with your domain terms:
```text
openshift
kubernetes
kubeflow
pipeline
container
automation
```

### Option B: Pass Words via Submission CLI
```bash
python pipeline/submit_pipeline.py --words "microservice,cloud,istio,telemetry"
```

### Option C: Pass Words File via Submission CLI
```bash
python pipeline/submit_pipeline.py --words-file my_custom_words.txt
```

---

## 🌐 Model Serving & API Protocols

The model is hosted using KServe and supports two simultaneous protocols:

### Friendly REST (`/complete`)
```bash
curl -X POST https://<ROUTE_HOST>/complete \
  -H "Content-Type: application/json" \
  -d '{"prefix": "ope"}'
```
Response:
```json
{
  "prefix": "ope",
  "completion": "openshift",
  "model": "char-completion",
  "source": "/mnt/models/model.pt"
}
```

### KServe v2 Data Plane (`/v2/models/char-completion/infer`)
```bash
curl -X POST https://<ROUTE_HOST>/v2/models/char-completion/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "prefix", "shape": [1], "datatype": "BYTES", "data": ["kube"]}
    ]
  }'
```

Interactive Swagger documentation is available at `https://<ROUTE_HOST>/docs`.

---

## 🧹 Teardown & Reset

To cleanly remove all cluster resources (DSPA, MinIO, KServe InferenceServices, Secrets, PVCs) and local artifacts:

```bash
python3 destroy.py
```
