import os
import sys
import time
import torch

def main():
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
            print(f"\n--- GPU {i}: {props.name} ---")
            print(f"  Compute Capability : {props.major}.{props.minor}")
            print(f"  Total Memory       : {props.total_memory / (1024 ** 2):.1f} MB")
            print(f"  Multi-Processors   : {props.multi_processor_count}")
        
        device = torch.device("cuda:0")
        print(f"\nRunning matrix multiplication test on device: {device}...")
        start_time = time.time()
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"Tensor shape result: {c.shape}")
        print(f"Computation completed in {elapsed * 1000:.2f} ms")
        print("\n✅ GPU Workload executed successfully!")
    else:
        print("\n⚠️ No GPU detected or running in CPU-only fallback mode.")
        device = torch.device("cpu")
        print(f"Running fallback matrix multiplication test on {device}...")
        start_time = time.time()
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = torch.matmul(a, b)
        elapsed = time.time() - start_time
        print(f"Tensor shape result: {c.shape}")
        print(f"Computation completed in {elapsed * 1000:.2f} ms")
        print("\n✅ CPU Fallback executed successfully!")

    print("=" * 60)

if __name__ == "__main__":
    main()
