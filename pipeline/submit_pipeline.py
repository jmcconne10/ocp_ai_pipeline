import argparse
import os
import subprocess
import sys
import urllib3
from pathlib import Path
from dotenv import load_dotenv
import kfp

# Load local .env if present
load_dotenv(override=False)

def get_ocp_token() -> str:
    """Retrieve token from environment variable or active `oc` session."""
    token = os.getenv("OCP_TOKEN", "").strip()
    if token:
        return token
    
    try:
        res = subprocess.run(["oc", "whoami", "-t"], capture_output=True, text=True, check=True)
        token = res.stdout.strip()
        if token:
            print("Retrieved OpenShift token from active CLI session (`oc whoami -t`).")
            return token
    except Exception:
        pass
    
    print("❌ Error: OpenShift authentication token not found.")
    print("Please set OCP_TOKEN in your .env file or log in with `oc login`.")
    sys.exit(1)


def get_dspa_host() -> str:
    """Retrieve DSPA Route URL from environment variable or active `oc` namespace."""
    host = os.getenv("DSPA_HOST", "").strip()
    if host:
        return host
    
    try:
        res = subprocess.run(
            ["oc", "get", "route", "-l", "app.kubernetes.io/name=data-science-pipelines",
             "-o", 'jsonpath={"https://"}{.items[0].spec.host}'],
            capture_output=True, text=True, check=True
        )
        host = res.stdout.strip()
        if host and host != "https://":
            print(f"Auto-discovered DSPA host from cluster route: {host}")
            return host
    except Exception:
        pass
    
    print("❌ Error: DSPA_HOST not specified.")
    print("Please set DSPA_HOST in your .env file.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Submit KFP v2 Pipeline to OpenShift AI")
    parser.add_argument(
        "--pipeline-file",
        "-f",
        default="pipeline.yaml",
        help="Path to compiled pipeline YAML file (default: pipeline.yaml)"
    )
    parser.add_argument(
        "--num-gpus",
        "-g",
        type=int,
        default=int(os.getenv("NUM_GPUS", "1")),
        help="Number of GPUs to request for the workload (set 0 for CPU-only trial clusters)"
    )
    parser.add_argument(
        "--experiment-name",
        "-e",
        default="gpu-pipeline-experiments",
        help="KFP Experiment name (default: gpu-pipeline-experiments)"
    )
    parser.add_argument(
        "--run-name",
        "-r",
        default=None,
        help="Custom run name (default: auto-generated timestamped name)"
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=os.getenv("VERIFY_SSL", "True").lower() in ("false", "0"),
        help="Disable SSL certificate verification for clusters with internal/self-signed certs"
    )
    
    args = parser.parse_args()

    # Compile pipeline if compiled file does not exist
    pipeline_path = Path(args.pipeline_file)
    if not pipeline_path.exists():
        print(f"Pipeline file {pipeline_path} not found. Compiling pipeline...")
        from pipeline import compile_pipeline
        compile_pipeline(str(pipeline_path))

    token = get_ocp_token()
    dspa_host = get_dspa_host()
    verify_ssl = not args.insecure

    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("⚠️ SSL verification disabled (self-signed certificate mode).")

    print(f"Connecting to OpenShift AI DSPA at: {dspa_host}")
    
    try:
        client = kfp.Client(
            host=dspa_host,
            existing_token=token,
            verify_ssl=verify_ssl
        )
        
        # 1. Register or update the pipeline template in OpenShift AI catalog
        pipeline_id = None
        try:
            print("Registering pipeline template in OpenShift AI UI catalog...")
            p = client.upload_pipeline(
                pipeline_package_path=str(pipeline_path),
                pipeline_name="ocp-gpu-hello-world"
            )
            pipeline_id = p.pipeline_id
            print(f"✅ Pipeline template registered: {pipeline_id}")
        except Exception:
            try:
                pipeline_id = client.get_pipeline_id(name="ocp-gpu-hello-world")
                if pipeline_id:
                    import time
                    version_name = f"v-{int(time.time())}"
                    client.upload_pipeline_version(
                        pipeline_package_path=str(pipeline_path),
                        pipeline_version_name=version_name,
                        pipeline_id=pipeline_id
                    )
                    print(f"✅ Uploaded new pipeline version '{version_name}' to catalog.")
            except Exception as upload_err:
                # Non-fatal if upload version fails; run can still execute
                pass

        # 2. Ensure experiment exists
        try:
            experiment = client.create_experiment(name=args.experiment_name)
            experiment_id = experiment.experiment_id
        except Exception:
            # If already exists, retrieve it
            experiment = client.get_experiment(experiment_name=args.experiment_name)
            experiment_id = experiment.experiment_id
            
        run_name = args.run_name or f"gpu-run-gpus-{args.num_gpus}"
        pipeline_params = {"num_gpus": args.num_gpus}
        
        print(f"Submitting pipeline run '{run_name}' (num_gpus={args.num_gpus})...")
        run_result = client.create_run_from_pipeline_package(
            pipeline_file=str(pipeline_path),
            arguments=pipeline_params,
            run_name=run_name,
            experiment_name=args.experiment_name
        )
        
        print("\n" + "=" * 60)
        print("🚀 Pipeline submitted successfully!")
        print("=" * 60)
        print(f"Run ID        : {run_result.run_id}")
        print(f"Experiment ID : {experiment_id}")
        print(f"Status Details: {dspa_host}/#/runs/details/{run_result.run_id}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Pipeline submission failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
