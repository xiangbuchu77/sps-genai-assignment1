"""Verify the running assignment container and save reproducible evidence.

Run from the repository root after building and starting the Docker container:
    python tools/verify_container.py
Only Python's standard library and the Docker CLI are required.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def inspect(container):
    return json.loads(subprocess.check_output(["docker", "inspect", container]))[0]


def verify(container, base_url):
    for _ in range(90):
        state = inspect(container)
        if not state["State"]["Running"]:
            raise RuntimeError("Container is not running.")
        if state["State"].get("Health", {}).get("Status") == "healthy":
            break
        time.sleep(1)
    else:
        raise RuntimeError("Container did not become healthy within 90 seconds.")
    assert state["Mounts"] == [], "Verification requires a self-contained image without mounts."

    cases = [
        ("GET", "/", None, 200),
        ("GET", "/docs", None, 200),
        ("GET", "/openapi.json", None, 200),
        ("POST", "/embedding", {"word": "apple"}, 200),
        ("POST", "/embedding", {"word": "   "}, 422),
        ("POST", "/embedding", {"word": "two words"}, 422),
        ("POST", "/embedding", {"word": "zzqzxqvnotawordzz"}, 404),
        ("POST", "/generate", {"start_word": "the", "length": 8}, 200),
    ]
    checks = []
    for method, path, payload, expected in cases:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status = response.status
            body = response.read().decode()
        assert status == expected, (path, status, body)
        if path == "/":
            assert json.loads(body) == {"Hello": "World"}
        if path == "/openapi.json":
            assert {"/", "/generate", "/embedding"}.issubset(json.loads(body)["paths"])
        if path == "/embedding" and expected == 200:
            actual = json.loads(body)
            assert actual["dimensions"] == len(actual["embedding"]) == 300
            reference = json.loads((ROOT / "evidence/embedding_apple.json").read_text())
            assert actual == reference, "Container output differs from the verified local model."
        if path == "/generate":
            words = json.loads(body)["generated_text"].split()
            assert words[0] == "the" and 1 <= len(words) <= 8
        checks.append({"method": method, "path": path, "payload": payload,
                       "expected_status": expected, "actual_status": status})

    image = json.loads(subprocess.check_output(["docker", "image", "inspect", state["Image"]]))[0]
    report = {
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": image["Os"] + "/" + image["Architecture"],
        "image_id": state["Image"],
        "container_name": container,
        "container_health": state["State"]["Health"]["Status"],
        "host_mounts": state["Mounts"],
        "build_verified": True,
        "container_http_verified": True,
        "apple_vector_matches_local_reference": True,
        "http_base_url": base_url,
        "http_checks": checks,
    }
    dest = ROOT / "evidence/docker_verification.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="sps-genai-assignment1")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    verify(args.container, args.base_url.rstrip("/"))
