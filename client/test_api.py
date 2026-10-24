#!/usr/bin/env python3
"""
API Test Client for Hosted Model
Tests both the Friendly REST endpoint and the KServe v2 Data Plane protocol.
Works against local servers (http://localhost:8080) or remote OpenShift routes.
"""

import argparse
import json
import os
import sys
import time
import urllib3
import requests
from dotenv import load_dotenv

load_dotenv()


def get_default_url() -> str:
    # 1. From CLI or environment
    env_url = os.getenv("SERVING_URL", "").strip()
    if env_url:
        return env_url

    # 2. Try discovering from OpenShift KServe or Route
    try:
        import subprocess
        # Check KServe route or standard route
        res = subprocess.run(
            ["oc", "get", "route", "char-completion", "-o", 'jsonpath={"https://"}{.spec.host}'],
            capture_output=True, text=True
        )
        url = res.stdout.strip()
        if url and url != "https://":
            return url
    except Exception:
        pass

    return "http://localhost:8080"


def test_health(base_url: str, verify_ssl: bool = True):
    url = f"{base_url.rstrip('/')}/health"
    print(f"\nChecking server health at: {url}")
    try:
        r = requests.get(url, timeout=5, verify=verify_ssl)
        print(f"  HTTP Status : {r.status_code}")
        print(f"  Response    : {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def call_friendly_api(base_url: str, prefix: str, verify_ssl: bool = True) -> str:
    url = f"{base_url.rstrip('/')}/complete"
    payload = {"prefix": prefix, "max_length": 25}
    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=10, verify=verify_ssl)
        elapsed = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json()
            completion = data.get("completion", "")
            print(f"  [REST] Input: '{prefix:<6}' -> Completed: '{completion}' ({elapsed:.1f}ms)")
            return completion
        else:
            print(f"  ❌ REST error ({r.status_code}): {r.text}")
            return ""
    except Exception as e:
        print(f"  ❌ Failed to connect to REST endpoint: {e}")
        return ""


def call_kserve_v2(base_url: str, prefix: str, verify_ssl: bool = True) -> str:
    url = f"{base_url.rstrip('/')}/v2/models/char-completion/infer"
    payload = {
        "inputs": [
            {
                "name": "prefix",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prefix]
            }
        ]
    }
    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=10, verify=verify_ssl)
        elapsed = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json()
            outputs = data.get("outputs", [])
            completion = outputs[0]["data"][0] if outputs and outputs[0].get("data") else ""
            print(f"  [KServe v2] Input: '{prefix:<6}' -> Completed: '{completion}' ({elapsed:.1f}ms)")
            return completion
        else:
            print(f"  ❌ KServe error ({r.status_code}): {r.text}")
            return ""
    except Exception as e:
        print(f"  ❌ Failed to connect to KServe endpoint: {e}")
        return ""


def run_interactive(base_url: str, verify_ssl: bool = True, use_kserve: bool = False):
    protocol = "KServe v2 Protocol" if use_kserve else "Friendly REST Protocol"
    print("\n" + "=" * 60)
    print(f"🤖 Interactive API Completion Client ({protocol})")
    print(f"Target Server: {base_url}")
    print("Type a prefix and press Enter (or 'exit' to quit).")
    print("=" * 60 + "\n")

    while True:
        try:
            p = input("API Prompt > ").strip()
            if not p:
                continue
            if p.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if use_kserve:
                call_kserve_v2(base_url, p, verify_ssl=verify_ssl)
            else:
                call_friendly_api(base_url, p, verify_ssl=verify_ssl)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Query Hosted Model API")
    parser.add_argument("--url", "-u", default=get_default_url(), help="Target server base URL (default: http://localhost:8080 or OpenShift route)")
    parser.add_argument("--prefix", "-p", default=None, help="Prefix to complete")
    parser.add_argument("--kserve", action="store_true", help="Use standard KServe v2 protocol instead of friendly REST")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive prompt loop")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification for self-signed cluster certs")
    args = parser.parse_args()

    verify_ssl = not args.insecure
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    base_url = args.url.rstrip("/")
    print("=" * 60)
    print(f"Model Serving API Client -> {base_url}")
    print("=" * 60)

    is_healthy = test_health(base_url, verify_ssl=verify_ssl)
    if not is_healthy:
        print("\n⚠️ Server is not responding. Ensure your serving container or KServe pod is running.")
        sys.exit(1)

    if args.interactive or not args.prefix:
        if not args.prefix:
            print("\nRunning Verification Test Requests:")
            print("-" * 50)
            sample_prefixes = ["ope", "kube", "py", "con", "kser"]
            for s in sample_prefixes:
                if args.kserve:
                    call_kserve_v2(base_url, s, verify_ssl=verify_ssl)
                else:
                    call_friendly_api(base_url, s, verify_ssl=verify_ssl)
            print("-" * 50)
            run_interactive(base_url, verify_ssl=verify_ssl, use_kserve=args.kserve)
        else:
            run_interactive(base_url, verify_ssl=verify_ssl, use_kserve=args.kserve)
    else:
        if args.kserve:
            call_kserve_v2(base_url, args.prefix, verify_ssl=verify_ssl)
        else:
            call_friendly_api(base_url, args.prefix, verify_ssl=verify_ssl)


if __name__ == "__main__":
    main()
