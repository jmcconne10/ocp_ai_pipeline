import argparse
from kfp import compiler, dsl

DEFAULT_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"

@dsl.container_component
def gpu_hello_world_op(num_gpus: int = 1):
    """
    Kubeflow Pipeline Container Component that runs PyTorch GPU device discovery
    and matrix multiplication operations.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["python", "-c"],
        args=[
            """
import sys, time, torch
print("=" * 60)
print("OpenShift AI Pipeline Workload: PyTorch Device Discovery")
print("=" * 60)
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"Detected GPU Devices: {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\\n--- GPU {i}: {props.name} ---")
        print(f"  Compute Capability : {props.major}.{props.minor}")
        print(f"  Total Memory       : {props.total_memory / (1024 ** 2):.1f} MB")
    device = torch.device("cuda:0")
    print(f"\\nRunning matrix multiplication test on {device}...")
    start = time.time()
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"Computation completed in {(time.time() - start) * 1000:.2f} ms")
    print("\\n[SUCCESS] GPU Workload executed successfully!")
else:
    print("\\n[INFO] No GPU detected or running in CPU fallback mode.")
    device = torch.device("cpu")
    print(f"Running fallback matrix multiplication test on {device}...")
    start = time.time()
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = torch.matmul(a, b)
    print(f"Computation completed in {(time.time() - start) * 1000:.2f} ms")
    print("\\n[SUCCESS] CPU Fallback executed successfully!")
print("=" * 60)
"""
        ]
    )

@dsl.pipeline(
    name="ocp-gpu-hello-world-pipeline",
    description="OpenShift AI Kubeflow Pipeline running GPU device discovery with CPU fallback support."
)
def gpu_pipeline(num_gpus: int = 1):
    task = gpu_hello_world_op(num_gpus=num_gpus)
    task.set_caching_options(False)
    
    # Configure GPU resources only if requested
    if num_gpus > 0:
        task.set_gpu_limit(num_gpus)
        task.set_accelerator_type("nvidia.com/gpu")
    
    task.set_cpu_request("500m")
    task.set_cpu_limit("2")
    task.set_memory_request("1Gi")
    task.set_memory_limit("4Gi")


def compile_pipeline(output_path: str = "pipeline.yaml"):
    print(f"Compiling pipeline to: {output_path}")
    compiler.Compiler().compile(
        pipeline_func=gpu_pipeline,
        package_path=output_path
    )
    print("Compilation successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile OpenShift AI GPU Pipeline")
    parser.add_argument(
        "--output",
        "-o",
        default="pipeline.yaml",
        help="Path for compiled pipeline YAML"
    )
    args = parser.parse_args()
    compile_pipeline(args.output)
