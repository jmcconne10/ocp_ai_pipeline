You are an expert MLOps and OpenShift platform engineer. 

I manage an OpenShift AI (RHOAI) environment. I am creating a reusable template git repository for our internal GitHub organization to help data scientists and developers run Kubeflow Pipelines from their local IDEs.

Please scaffold this project completely in the current directory, adhering to the following requirements:

### 1. Goal & Functional Requirements
- **Target Platform:** OpenShift AI (Data Science Pipelines 2.x / KFP v2 SDK).
- **Core Pipeline:** A Kubeflow Pipeline that runs a containerized "Hello World" Python script utilizing a GPU (`torch.cuda.is_available()`, device discovery, and a basic tensor operation).
- **GPU Configuration:** Include resource requests/limits (`nvidia.com/gpu: 1`) and common OpenShift GPU node tolerations.
- **Execution Modes:**
  1. **Manual / Local CLI Trigger:** A Python utility script (`submit_pipeline.py`) using `kfp.Client` so developers can submit and run the pipeline against the OpenShift Data Science Pipelines Application (DSPA) endpoint using an active `oc` token or environment variables.
  2. **Automated CI/CD Trigger:** A GitHub Actions workflow (`.github/workflows/pipeline_run.yaml`) that triggers on push/PR, compiles the pipeline, and submits it to OpenShift AI using repository secrets (`OPENSHIFT_SERVER`, `OPENSHIFT_TOKEN`, `DSPA_ROUTE_URL`).

### 2. Required Directory Structure
Create all necessary directories and files:
- `src/main.py`: The GPU hello world script.
- `src/requirements.txt`: Python dependencies for the workload container (e.g., PyTorch).
- `pipeline/pipeline.py`: The KFP v2 pipeline definition and compilation logic.
- `pipeline/requirements.txt`: Local/CI dependencies (`kfp>=2.0.0`, `kfp-server-api`).
- `pipeline/submit_pipeline.py`: The submission script that authenticates via bearer token to the DSPA route and starts an experiment run.
- `.github/workflows/pipeline_run.yaml`: GitHub Actions CI definition for compilation and remote submission.
- `Dockerfile`: Container image definition to run `src/main.py` using a standard OpenShift AI base (or CUDA runtime).
- `.env.example`: Template for required local environment variables (`DSPA_HOST`, `OCP_TOKEN`, etc.).
- `.gitignore`: Standard Python and KFP ignore rules (virtualenvs, compiled pipeline YAML/tarballs, `.env`).
- `README.md`: Comprehensive, step-by-step developer guide explaining:
  1. How to log in via `oc` and retrieve your DSPA endpoint URL and token.
  2. How to set up a local Python virtual environment.
  3. How to compile the pipeline locally to YAML.
  4. How to submit runs manually via the CLI script.
  5. How to configure GitHub repository secrets for automated commit-based runs.

### 3. Verification & Execution Steps
After writing the files:
1. Provide the exact `oc` commands I should run in the terminal to:
   - Identify the DSPA route in my trial namespace.
   - Extract my current login token for testing.
   - Check if GPU nodes and drivers exist on my cluster.
2. Ensure `submit_pipeline.py` has a fallback/parameter to run cleanly with `0` GPUs for trial clusters that lack physical NVIDIA hardware.