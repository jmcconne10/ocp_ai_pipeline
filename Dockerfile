# Base image with CUDA runtime and PyTorch pre-installed
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install additional requirements if specified
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source workload
COPY src/main.py /app/main.py

# OpenShift arbitrary user ID support (runs as non-root)
RUN chgrp -R 0 /app && \
    chmod -R g=u /app

USER 1001

ENTRYPOINT ["python", "/app/main.py"]
