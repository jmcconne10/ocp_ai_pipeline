#!/usr/bin/env python3
"""
OpenShift AI Pipeline Setup Script
Automates cluster preparation, MinIO/S3 configuration, DSPA deployment,
KServe serving configuration, route discovery, and .env generation.
"""

import os
import re
import sys
import time
import subprocess
import getpass
from pathlib import Path

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def run_cmd(cmd: list, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture
    )

def check_prerequisites():
    print_header("1. Checking Prerequisites")
    try:
        res = run_cmd(["which", "oc"])
        print(f"  [OK] OpenShift CLI found at: {res.stdout.strip()}")
    except Exception:
        print("  [ERROR] 'oc' CLI binary not found in your PATH.")
        print("  Please install the OpenShift CLI first: https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/")
        sys.exit(1)


def authenticate_openshift():
    print_header("2. OpenShift Authentication")
    
    current_user = None
    try:
        res = run_cmd(["oc", "whoami"])
        current_user = res.stdout.strip()
    except Exception:
        current_user = None

    if current_user:
        try:
            token_res = run_cmd(["oc", "whoami", "-t"])
            active_token = token_res.stdout.strip()
            use_current = input(f"Detected active session as '{current_user}'. Use this session? [Y/n]: ").strip().lower()
            if use_current in ("", "y", "yes"):
                return active_token
        except Exception:
            pass

    print("\nPlease provide your OpenShift cluster connection details.")
    print("Tip: You can get your token from the OpenShift Web Console -> top-right profile icon -> 'Copy login command'.\n")
    
    server = input("OpenShift API Server URL (e.g. https://api.mycluster.domain:6443): ").strip()
    while not server:
        server = input("Server URL cannot be empty. Please enter URL: ").strip()

    token = getpass.getpass("OpenShift Token (sha256~...): ").strip()
    while not token:
        token = getpass.getpass("Token cannot be empty. Please enter token: ").strip()

    print(f"\nLogging in to {server}...")
    try:
        run_cmd(["oc", "login", f"--server={server}", f"--token={token}"])
        user_res = run_cmd(["oc", "whoami"])
        print(f"  [OK] Successfully logged in as: {user_res.stdout.strip()}")
        return token
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Login failed: {e.stderr if e.stderr else 'Invalid credentials'}")
        sys.exit(1)


def get_namespace() -> str:
    print_header("3. Project / Namespace Selection")
    try:
        curr_ns = run_cmd(["oc", "project", "-q"]).stdout.strip()
    except Exception:
        curr_ns = ""

    prompt_msg = f"Target namespace [{curr_ns}]: " if curr_ns else "Target namespace: "
    ns = input(prompt_msg).strip() or curr_ns
    
    while not ns:
        ns = input("Namespace cannot be empty: ").strip()

    run_cmd(["oc", "project", ns])
    print(f"  [OK] Using project: {ns}")
    return ns


def verify_rhoai_operator():
    print("\nVerifying OpenShift AI (RHOAI) Data Science Pipelines Operator...")
    try:
        res = run_cmd(["oc", "get", "crd", "datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io"])
        print("  [OK] OpenShift AI Data Science Pipelines Operator is installed.")
    except Exception:
        print("  [WARNING] Data Science Pipelines CRD not found.")
        print("  Ensure Red Hat OpenShift AI operator is installed on your cluster.")


def configure_storage(namespace: str):
    print_header("4. Object Storage Configuration")
    print("Kubeflow Pipelines requires an S3-compatible bucket for artifacts.")
    print("  [1] In-Cluster MinIO (Self-contained, 100% free, ideal for trials & sandboxes)")
    print("  [2] External S3 (AWS S3, corporate Ceph, MinIO, or OpenShift Data Foundation)")
    
    choice = input("\nChoose storage option [1]: ").strip() or "1"
    
    if choice == "1":
        print("\nDeploying In-Cluster MinIO storage...")
        run_cmd(["oc", "apply", "-f", "manifests/minio.yaml", "-n", namespace])
        
        # Apply NetworkPolicy in case cluster is hardened
        if Path("manifests/networkpolicy-minio.yaml").exists():
            print("Applying NetworkPolicy for operator health-checks...")
            run_cmd(["oc", "apply", "-f", "manifests/networkpolicy-minio.yaml", "-n", namespace])
            
        print("Deploying DataSciencePipelinesApplication (DSPA)...")
        run_cmd(["oc", "apply", "-f", "manifests/dspa.yaml", "-n", namespace])
        print("  [OK] MinIO and DSPA manifests applied successfully.")
    else:
        print("\n--- External S3 Configuration ---")
        host = input("S3 Host (e.g. s3.amazonaws.com or internal Ceph host): ").strip()
        bucket = input("Bucket name: ").strip()
        region = input("Region [us-east-1]: ").strip() or "us-east-1"
        scheme = input("Scheme (https or http) [https]: ").strip() or "https"
        access_key = input("Access Key ID: ").strip()
        secret_key = getpass.getpass("Secret Access Key: ").strip()
        
        run_cmd([
            "oc", "create", "secret", "generic", "pipeline-s3-secret",
            f"--from-literal=accesskey={access_key}",
            f"--from-literal=secretkey={secret_key}",
            "-n", namespace,
            "--dry-run=client", "-o", "yaml", "|", "oc", "apply", "-f", "-"
        ], check=False)
        
        print("\nCreating DSPA referencing external S3...")
        dspa_external = f"""apiVersion: datasciencepipelinesapplications.opendatahub.io/v1
kind: DataSciencePipelinesApplication
metadata:
  name: dspa
spec:
  dspVersion: v2
  database:
    mariaDB:
      deploy: true
      pvcSize: 10Gi
  objectStorage:
    disableHealthCheck: false
    enableExternalRoute: false
    externalStorage:
      basePath: ""
      bucket: {bucket}
      host: {host}
      port: "{"443" if scheme == "https" else "80"}"
      region: {region}
      s3CredentialsSecret:
        accessKey: accesskey
        secretKey: secretkey
        secretName: pipeline-s3-secret
      scheme: {scheme}
      secure: {"true" if scheme == "https" else "false"}
"""
        with open("/tmp/dspa-external.yaml", "w") as f:
            f.write(dspa_external)
        run_cmd(["oc", "apply", "-f", "/tmp/dspa-external.yaml", "-n", namespace])
        print("  [OK] External S3 DSPA applied.")


def configure_kserve_serving(namespace: str):
    print_header("5. Configuring Model Serving & KServe RBAC")
    
    # 1. Grant pipeline-runner-dspa permissions to manage KServe resources
    print("Configuring RBAC for pipeline ServiceAccount ('pipeline-runner-dspa')...")
    run_cmd(["oc", "adm", "policy", "add-role-to-user", "edit", "-z", "pipeline-runner-dspa", "-n", namespace], check=False)
    print("  [OK] RBAC configured.")

    # 2. Apply Serving manifests (Secret, ServingRuntime, BuildConfig)
    print("\nConfiguring KServe S3 Secret & ServingRuntime...")
    if Path("serving/kserve-secret.yaml").exists():
        run_cmd(["oc", "apply", "-f", "serving/kserve-secret.yaml", "-n", namespace])
    if Path("serving/servingruntime.yaml").exists():
        run_cmd(["oc", "apply", "-f", "serving/servingruntime.yaml", "-n", namespace])
    if Path("serving/buildconfig.yaml").exists():
        run_cmd(["oc", "apply", "-f", "serving/buildconfig.yaml", "-n", namespace])
    print("  [OK] Model serving manifests applied.")

    # 3. Optional container image build on OpenShift
    do_build = input("\nBuild serving container image in OpenShift now? [Y/n]: ").strip().lower()
    if do_build in ("", "y", "yes"):
        print("Starting OpenShift binary build from serving/ ...")
        run_cmd(["oc", "start-build", "char-completion-serving", "--from-dir=serving", "--follow", "-n", namespace], capture=False)
        print("  [OK] Serving container image built.")


def wait_for_dspa_route(namespace: str) -> str:
    print_header("6. Waiting for DSPA Pipeline Server Route")
    print("Waiting for OpenShift AI to reconcile DSPA and create ingress route...")
    
    max_wait = 120
    start = time.time()
    while time.time() - start < max_wait:
        # 1. Try direct route name 'ds-pipeline-dspa'
        try:
            res = run_cmd([
                "oc", "get", "route", "ds-pipeline-dspa",
                "-n", namespace,
                "-o", 'jsonpath={"https://"}{.spec.host}'
            ])
            route = res.stdout.strip()
            if route and route != "https://" and "apps." in route:
                print(f"\n  [OK] DSPA Route is ready: {route}")
                return route
        except Exception:
            pass

        # 2. Search routes for any beginning with ds-pipeline- (excluding metadata route)
        try:
            res = run_cmd([
                "oc", "get", "routes", "-n", namespace,
                "-o", "jsonpath={range .items[*]}{.metadata.name}{' '}{.spec.host}{'\\n'}{end}"
            ])
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    name, host = parts[0], parts[1]
                    if name.startswith("ds-pipeline-") and not name.startswith("ds-pipeline-md-"):
                        route = f"https://{host}"
                        print(f"\n  [OK] Discovered DSPA Route '{name}': {route}")
                        return route
        except Exception:
            pass

        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(3)
        
    print("\n  [WARNING] Timed out waiting for automatic route creation.")
    print("  Check pod status with: oc get pods -n " + namespace)
    fallback = input("If you have the route URL, paste it here (or press Enter to skip): ").strip()
    return fallback


def check_gpu_quota(namespace: str) -> int:
    print_header("7. GPU Resource Quota Detection")
    num_gpus = 1
    try:
        res = run_cmd(["oc", "get", "resourcequota", "-n", namespace, "-o", "yaml"])
        if 'limits.nvidia.com/gpu: "0"' in res.stdout or 'requests.nvidia.com/gpu: "0"' in res.stdout:
            print("  [INFO] Detected namespace resource quota restricts GPU: limits.nvidia.com/gpu=0")
            print("  Configuring NUM_GPUS=0 for CPU fallback mode.")
            return 0
    except Exception:
        pass

    user_val = input("Number of GPUs for pipeline tasks [1]: ").strip() or "1"
    try:
        return int(user_val)
    except ValueError:
        return 1


def write_env_file(token: str, dspa_host: str, namespace: str, num_gpus: int):
    print_header("8. Writing Local Configuration (.env)")
    env_content = f"""# OpenShift & Data Science Pipelines Configuration
# Generated by setup.py

# 1. Your OpenShift Authentication Token:
OCP_TOKEN={token}

# 2. DSPA (Data Science Pipelines Application) Route URL:
DSPA_HOST={dspa_host}

# 3. OpenShift Project / Namespace where DSPA is running:
NAMESPACE={namespace}

# 4. Set to False if using trial / self-signed cluster certificates:
VERIFY_SSL=True

# 5. Number of GPUs for pipeline tasks (0 for CPU fallback):
NUM_GPUS={num_gpus}
"""
    with open(".env", "w") as f:
        f.write(env_content)
    print("  [OK] Updated .env file successfully.")


def setup_virtualenv_and_compile():
    print_header("9. Local Python Environment & Pipeline Compilation")
    do_compile = input("Set up virtualenv (.venv) and pre-compile the pipeline now? [Y/n]: ").strip().lower()
    if do_compile not in ("", "y", "yes"):
        print("Skipping local compilation.")
        return

    venv_dir = Path(".venv")
    if not venv_dir.exists():
        print("Creating Python virtual environment in .venv...")
        run_cmd([sys.executable, "-m", "venv", ".venv"])
    
    pip_bin = str(venv_dir / "bin" / "pip")
    py_bin = str(venv_dir / "bin" / "python")
    
    print("Installing requirements from pipeline/requirements.txt...")
    run_cmd([pip_bin, "install", "--upgrade", "pip"], capture=False)
    run_cmd([pip_bin, "install", "-r", "pipeline/requirements.txt"], capture=False)
    
    print("\nCompiling pipeline.py to pipeline.yaml...")
    run_cmd([py_bin, "pipeline/pipeline.py", "--output", "pipeline.yaml"], capture=False)
    print("  [OK] pipeline.yaml compiled successfully.")

    do_upload = input("\nSubmit and upload initial run to OpenShift AI now? [Y/n]: ").strip().lower()
    if do_upload in ("", "y", "yes"):
        print("\nSubmitting pipeline run...")
        run_cmd([py_bin, "pipeline/submit_pipeline.py"], capture=False)


def main():
    print("=" * 65)
    print("  OpenShift AI Pipeline Setup Wizard")
    print("=" * 65)
    print("This script configures your OpenShift namespace, storage,")
    print("KServe model serving, DSPA pipeline server, and local .env settings.\n")

    check_prerequisites()
    token = authenticate_openshift()
    namespace = get_namespace()
    verify_rhoai_operator()
    configure_storage(namespace)
    configure_kserve_serving(namespace)
    dspa_host = wait_for_dspa_route(namespace)
    num_gpus = check_gpu_quota(namespace)
    write_env_file(token, dspa_host, namespace, num_gpus)
    setup_virtualenv_and_compile()

    print_header("Setup Completed Successfully!")
    print("Next steps:")
    print("  1. Activate virtual environment:  source .venv/bin/activate")
    print("  2. Submit pipeline runs anytime: python pipeline/submit_pipeline.py")
    print("  3. Test model API:               python client/test_api.py")
    print("  4. Check ARCHITECTURE_EXPLANATION.md for enterprise replication tips.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup aborted by user.")
        sys.exit(1)
