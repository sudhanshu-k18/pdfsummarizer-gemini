"""Phase 0 environment check: Python, dependencies, Gemini API key and GitHub CLI.

Usage:
    .venv\\Scripts\\python.exe scripts\\check_environment.py
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "langchain-core": "langchain_core",
    "langchain-google-genai": "langchain_google_genai",
    "pypdf": "pypdf",
    "python-dotenv": "dotenv",
    "pandas": "pandas",
    "tiktoken": "tiktoken",
}

failures: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    if not ok:
        failures.append(label)


def clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


def check_python() -> None:
    version = sys.version.split()[0]
    report(sys.version_info >= (3, 10), "Python >= 3.10", version)


def load_env() -> None:
    env_path = ROOT / ".env"
    report(env_path.exists(), ".env file exists", str(env_path))
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "python-dotenv available", "run: pip install -r requirements.txt")
        return
    load_dotenv(env_path)


def check_packages() -> None:
    for dist, _module in REQUIRED_PACKAGES.items():
        try:
            version = importlib.metadata.version(dist)
            report(True, f"{dist} installed", version)
        except importlib.metadata.PackageNotFoundError:
            report(False, f"{dist} installed", "missing")


def gemini_request(path: str, api_key: str, payload: dict | None = None, timeout: int = 30):
    url = f"https://generativelanguage.googleapis.com/v1beta/{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"x-goog-api-key": api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            message = body
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc


def check_gemini() -> None:
    api_key = clean(os.getenv("GEMINI_API_KEY"))
    report(bool(api_key), "GEMINI_API_KEY set")
    if not api_key:
        return

    try:
        data = gemini_request("models", api_key)
    except Exception as exc:  # noqa: BLE001
        report(False, "Gemini API key accepted", str(exc))
        return

    models = [
        m["name"].removeprefix("models/")
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    report(bool(models), "Gemini API key accepted", f"{len(models)} text models available")

    model = clean(os.getenv("GEMINI_MODEL")) or "gemini-3.1-flash-lite"
    if models and model not in models:
        report(False, "Configured model available", f"{model} is not in the model list")
        return
    report(True, "Configured model available", model)

    try:
        data = gemini_request(
            f"models/{model}:generateContent",
            api_key,
            {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]},
        )
    except Exception as exc:  # noqa: BLE001
        report(False, f"Model {model} responded", str(exc))
        return

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    reply = " ".join(p.get("text", "") for p in parts).strip()
    report(bool(reply), f"Model {model} responded", reply or "empty response")


def find_gh() -> str | None:
    gh = shutil.which("gh")
    if gh:
        return gh
    fallback = Path(r"C:\Program Files\GitHub CLI\gh.exe")
    return str(fallback) if fallback.exists() else None


def check_github() -> None:
    gh = find_gh()
    report(bool(gh), "GitHub CLI installed", gh or "winget install --id GitHub.cli")
    if not gh:
        return

    proc = subprocess.run([gh, "auth", "status"], capture_output=True, text=True)
    output = (proc.stdout or proc.stderr).strip()
    first_line = output.splitlines()[0] if output else "no output"
    report(proc.returncode == 0, "GitHub CLI authenticated", first_line)

    if proc.returncode == 0:
        user = subprocess.run(
            [gh, "api", "user", "-q", ".login"], capture_output=True, text=True
        )
        if user.returncode == 0:
            report(True, "GitHub user resolved", user.stdout.strip())


def main() -> int:
    print("=" * 60)
    print("Phase 0 environment check")
    print("=" * 60)

    check_python()
    load_env()
    check_packages()
    check_gemini()
    check_github()

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} check(s) failed")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
