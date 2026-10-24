# Configurable base image (can be pointed to company internal registry mirror)
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py train.py /app/

# OpenShift arbitrary user ID support
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001

ENTRYPOINT ["python", "/app/train.py"]
