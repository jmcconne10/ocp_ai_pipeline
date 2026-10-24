# Client & Inference Testing

This module provides tools to test and interact with the character completion model, both locally and against the hosted OpenShift AI endpoint.

---

## 🚀 Quick Usage

### 1. Test Hosted Model API via CLI (`test_api.py`)

The test client automatically detects your active OpenShift route or connects to `http://localhost:8080`.

```bash
# Friendly REST API (Interactive Prompt)
python client/test_api.py

# Single query via Friendly REST
python client/test_api.py --prefix "ope"

# Enterprise KServe v2 Data Plane protocol
python client/test_api.py --kserve --prefix "kube"

# Target a specific OpenShift Route or URL
python client/test_api.py --url https://char-completion-joe-mcconnell10-dev.apps.mycluster.domain --prefix "py"

# Insecure mode (for clusters with internal/self-signed certs)
python client/test_api.py --insecure --prefix "ten"
```

---

## 🌐 API Protocols

The model service supports two standard protocols simultaneously:

### Protocol 1: Friendly REST API (`/complete`)

Designed for easy developer testing, browser inspection, and curl scripts.

#### Request:
```bash
curl -X POST https://<ROUTE_HOST>/complete \
  -H "Content-Type: application/json" \
  -d '{"prefix": "ope", "max_length": 25}'
```

#### Response:
```json
{
  "prefix": "ope",
  "completion": "openshift",
  "model": "char-completion",
  "source": "/mnt/models/model.pt"
}
```

---

### Protocol 2: KServe v2 Data Plane (`/v2/models/char-completion/infer`)

The enterprise standard protocol implemented by Triton, TorchServe, vLLM, and OpenVINO Model Server.

#### Request:
```bash
curl -X POST https://<ROUTE_HOST>/v2/models/char-completion/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prefix",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["kube"]
      }
    ]
  }'
```

#### Response:
```json
{
  "model_name": "char-completion",
  "model_version": "v1",
  "id": null,
  "outputs": [
    {
      "name": "completion",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["kubernetes"]
    }
  ]
}
```

---

## 🩺 Health & Readiness Probes

```bash
# Liveness / Readiness
curl https://<ROUTE_HOST>/health
curl https://<ROUTE_HOST>/v2/health/ready

# Model Metadata
curl https://<ROUTE_HOST>/v2/models/char-completion

# Interactive Swagger Documentation
# Open https://<ROUTE_HOST>/docs in any web browser
```

---

## 💻 Python Client Snippet

```python
import requests

ROUTE_URL = "https://<ROUTE_HOST>"

# Option A: Friendly REST
response = requests.post(f"{ROUTE_URL}/complete", json={"prefix": "con"})
print("Completion:", response.json()["completion"])

# Option B: KServe v2 Data Plane
payload = {
    "inputs": [{"name": "prefix", "shape": [1], "datatype": "BYTES", "data": ["ten"]}]
}
res = requests.post(f"{ROUTE_URL}/v2/models/char-completion/infer", json=payload)
print("KServe output:", res.json()["outputs"][0]["data"][0])
```
