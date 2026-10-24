# Project TODO & Roadmap

This document tracks completed features, architectural modularization, and company onboarding improvements for the OpenShift AI Pipeline template.

---

## 📋 Task List

### 1. Company Network & Enterprise Readiness
- [x] **Review and adapt all code for enterprise network restrictions**
  - Configurable `BASE_IMAGE` arguments in Dockerfiles to support internal corporate registry mirrors.
  - Proxy and self-signed certificate handling via `VERIFY_SSL` and `urllib3` warning suppression across all scripts (`setup.py`, `submit_pipeline.py`, `infer.py`, `client/test_api.py`).
  - Graceful S3 handling for internal MinIO/Ceph clusters with S3v4 signatures and disabled SSL verification options.
  - Sanitized log outputs (ASCII tags `[SUCCESS]`, `[INFO]`) to avoid MariaDB 3-byte `utf8` collation errors on OpenShift DSPA.

---

### 2. Modular Architecture (Separate Folders for Each Component)
- [x] **Restructure repository into standalone, decoupled modules so teams can adopt pieces independently:**
  - `data/`: Domain vocabulary dataset (`words.txt`).
  - `training/`: PyTorch `CharLSTM` model, vocabulary builders, standalone training script (`train.py`), Dockerfile, and requirements.
  - `pipeline/`: Kubeflow Pipelines (KFP v2) 3-step pipeline (`pipeline.py`), CLI submission tool (`submit_pipeline.py`), requirements.
  - `serving/`: Production-ready FastAPI serving application (`app.py`), Dockerfile, KServe `InferenceService`, `ServingRuntime`, `Secret`, and OpenShift `BuildConfig`.
  - `client/`: API test client (`test_api.py`) supporting both Friendly REST and KServe v2 protocols with interactive prompts.
  - `manifests/`: Infrastructure manifests (MinIO, DSPA, NetworkPolicies).
  - `tekton/`: Native OpenShift CI/CD manifests (Tasks, Pipelines, PipelineRuns).

---

### 3. Model Hosting
- [x] **Add ability to host the trained model on OpenShift**
  - KServe v1beta1 `InferenceService` manifest (`serving/inferenceservice.yaml`) configured for `RawDeployment` mode.
  - Custom `ServingRuntime` manifest (`serving/servingruntime.yaml`) for character completion models.
  - Automated Step 3 in Kubeflow Pipelines (`deploy_kserve_op`) to auto-reconcile Secrets, ServingRuntimes, and InferenceServices.
  - OpenShift `BuildConfig` and `ImageStream` (`serving/buildconfig.yaml`) for building serving container images on-cluster.
  - Fixed S3 model export path (`s3://mlpipeline/models/char-completion/model.pt`).

---

### 4. API Client & Test Calls
- [x] **Add ability to make API calls to the hosted model**
  - Dual-protocol support in `serving/app.py`:
    - Friendly REST endpoint (`/complete`) for rapid curl and browser testing.
    - KServe v2 Data Plane protocol (`/v2/models/char-completion/infer`) for enterprise standard compatibility.
  - Interactive CLI test client (`client/test_api.py`) with automatic OpenShift route detection.
  - Built-in Swagger API documentation UI (`/docs`).
  - Complete documentation and curl examples in `client/README.md`.

---

### 5. Customizable Training Vocabulary
- [x] **Add ability to change the words used for training dynamically**
  - Default domain vocabulary file in `data/words.txt`.
  - Training script support for `--words` (comma-separated) and `--words-file`.
  - Pipeline parameter `words_list: str` passed through `pipeline.py` and `submit_pipeline.py`.
  - Dynamic prefix evaluation based on active vocabulary words.
