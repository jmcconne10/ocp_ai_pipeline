# OpenShift AI Pipeline: Architecture & Enterprise Replication Guide

This document explains the technical architecture implemented in this repository, how each component functions, and the exact steps required to replicate this setup inside your company's enterprise OpenShift environment.

---

## 1. What We Built & How It Works

### The Core Problem
Data science teams need to run machine learning pipelines (preprocessing, training, evaluation) on OpenShift AI (RHOAI). However:
- Running pipelines from local laptops or IDEs often encounters authentication, TLS certificate, or S3 storage hurdles.
- Multi-tenant enterprise clusters often restrict cluster-wide permissions, require internal NetworkPolicies, and enforce GPU quotas.
- Many organizations do not have external CI/CD tools like GitHub Actions connected to their private clusters.

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
│  │                                                                          │  │
│  │  - OpenShift OAuth Proxy Ingress Route (`ds-pipeline-dspa`)              │  │
│  │  - KFP API Server (Validates pipeline specs & schedules runs)            │  │
│  │  - MariaDB (Tracks pipeline runs, experiments, and metadata)             │  │
│  │  - Argo Workflow Controller (Executes the pipeline Directed Acyclic Graph)│
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Object Storage (Pipeline Artifacts & Logs)                               │  │
│  │  - Trial: In-cluster MinIO pod + PVC                                     │  │
│  │  - Production: Company AWS S3 / Ceph / OpenShift Data Foundation         │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Workload Execution Pod                                                   │  │
│  │  - GPU Mode: `nvidia.com/gpu: 1` + PyTorch CUDA execution                │  │
│  │  - CPU Fallback: `num_gpus: 0` for development and non-GPU testing       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Components Explained

| Component | File in Repo | Purpose |
|---|---|---|
| **Workload Code** | `src/main.py` | PyTorch script that auto-detects CUDA hardware, inspects GPU properties, runs matrix multiplication, and falls back to CPU if no GPU is available. |
| **Pipeline Definition** | `pipeline/pipeline.py` | Written in **Kubeflow Pipelines v2 SDK** (`kfp>=2.7.0`). Defines component specs, container image, CPU/memory limits, and parameterizes GPU requests (`num_gpus`). Compiles to `pipeline.yaml`. |
| **CLI Runner** | `pipeline/submit_pipeline.py` | Submits `pipeline.yaml` to the remote DSPA route using `kfp.Client`. Automatically extracts your active OpenShift token (`oc whoami -t`) and route URL if not explicitly provided. |
| **Storage (Trial)** | `manifests/minio.yaml` | Deploys an S3-compatible MinIO instance with persistent storage (`gp3`) inside your project so no external AWS account is needed. |
| **Pipeline Server** | `manifests/dspa.yaml` | The OpenShift AI Custom Resource (`DataSciencePipelinesApplication`) that configures the pipeline server, metadata DB, and artifact storage. |
| **Network Security** | `manifests/networkpolicy-minio.yaml` | Grants the OpenShift AI Operator permission to health-check MinIO in hardened multi-tenant clusters. |
| **CI/CD (Native)** | `tekton/` | OpenShift Pipelines manifests to clone your internal git repo and run compilation/submission inside OpenShift without external CI tools. |

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

#### If your company provides an S3 bucket (AWS S3, Ceph, MinIO):
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

#### If your company cluster does NOT provide S3 (Use In-Cluster MinIO):
Simply run:
```bash
oc apply -f manifests/minio.yaml
oc apply -f manifests/networkpolicy-minio.yaml
oc apply -f manifests/dspa.yaml
```

### Step 3: Get Your DSPA Route & Configure `.env`
Once the DSPA pods are running (`oc get pods -l app.kubernetes.io/name=data-science-pipelines`):
```bash
# Get your DSPA route
oc get route -l app.kubernetes.io/name=data-science-pipelines -o jsonpath='{"https://"}{.items[0].spec.host}{"\n"}'
```
Copy `.env.example` to `.env` and fill in:
- `OCP_TOKEN`: Your token (`oc whoami -t`)
- `DSPA_HOST`: The route URL from above
- `NAMESPACE`: Your team project
- `NUM_GPUS`: `1` (if GPU nodes exist) or `0` (for CPU testing)

### Step 4: Run the Pipeline Locally
```bash
# Set up Python virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r pipeline/requirements.txt

# Compile and submit
python pipeline/submit_pipeline.py
```

### Step 5: (Optional) Set Up In-Cluster CI/CD with Tekton
If you want automatic runs when code changes in your internal Git server:
```bash
oc apply -f tekton/rbac.yaml
oc apply -f tekton/task-compile-submit.yaml
oc apply -f tekton/pipeline.yaml
oc create -f tekton/pipelinerun.yaml
```

---

## 4. Key Lessons Learned & Enterprise Gotchas

During the initial deployment, several subtle enterprise constraints were uncovered and solved:

1. **4-Byte Emojis in Pipeline Scripts vs MariaDB Charset:**
   - *Problem:* OpenShift AI's bundled MariaDB metadata database defaults to 3-byte `utf8` rather than `utf8mb4`. Any 4-byte unicode emojis (`✅`, `⚠️`, etc.) in pipeline code or component specs cause MySQL error `1366: Incorrect string value` when storing the pipeline manifest.
   - *Resolution:* All workload scripts and component definitions in this template use standard ASCII tags (e.g. `[SUCCESS]`, `[INFO]`).
2. **Multi-Tenant NetworkPolicies on MinIO:**
   - *Problem:* In secured OpenShift clusters (like Developer Sandbox or hardened corporate environments), inter-namespace traffic from the Operator namespace (`redhat-ods-applications`) to user namespaces is blocked by default. This causes the Operator's MinIO health-check to time out.
   - *Resolution:* Included `manifests/networkpolicy-minio.yaml` to explicitly allow the Operator pod to communicate with the MinIO port (9000).
3. **GPU Quotas on Non-GPU Clusters:**
   - *Problem:* Trial or developer projects often have cluster resource quotas setting `limits.nvidia.com/gpu: "0"`. Submitting a GPU request causes immediate Kubernetes scheduling rejection.
   - *Resolution:* `submit_pipeline.py` supports `--num-gpus 0`, and `pipeline/pipeline.py` dynamically strips GPU resource limits and node selectors when `num_gpus == 0`.
