ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/

# Support OpenShift arbitrary non-root user IDs (group 0 / root group per OpenShift standards)
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001

EXPOSE 8080

ENV PORT=8080
ENV MODEL_DIR=/mnt/models

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
