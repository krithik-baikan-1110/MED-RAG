#!/usr/bin/env python3
"""Convenience launcher for the MED-RAG project.

Runs both the FastAPI backend (on port 8000) and the Vite frontend (on port 5173)
so the whole stack comes up with a single command.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

IS_WINDOWS = os.name == "nt"
VENV_DIR = ROOT_DIR / "myenv"
VENV_SCRIPTS_DIR = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
VENV_PYTHON = VENV_SCRIPTS_DIR / ("python.exe" if IS_WINDOWS else "python")

DEFAULT_WEAVIATE_READY_URL = "http://localhost:8080/v1/.well-known/ready"

_PROCESSES: list[subprocess.Popen[bytes]] = []
_VIRTUALENV_INFO: tuple[str, Path, Path] | None = None


def _ensure_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"Expected {description} at {path}, but it was not found.")


def _command_for(executable: str) -> list[str]:
    candidate = shutil.which(executable)
    if candidate:
        return [candidate]

    if sys.platform.startswith("win"):
        candidate = shutil.which(f"{executable}.cmd") or shutil.which(f"{executable}.exe")
        if candidate:
            return [candidate]

    raise SystemExit(
        f"Required command '{executable}' not found on PATH. Please install it and retry."
    )


def _resolve_virtualenv_info() -> tuple[str, Path, Path]:
    override = os.environ.get("VENV_PYTHON")
    if override:
        python_path = Path(override).expanduser().resolve()
        if not python_path.exists():
            raise SystemExit(
                f"VENV_PYTHON was set to {python_path}, but that interpreter does not exist."
            )
        scripts_dir = python_path.parent
        venv_dir = scripts_dir.parent
        return str(python_path), scripts_dir, venv_dir

    python_path = VENV_PYTHON
    if not python_path.exists():
        raise SystemExit(
            f"Expected virtual environment Python at {python_path}. "
            "Create it with `python -m venv myenv` and install dependencies."
        )

    return str(python_path), VENV_SCRIPTS_DIR, VENV_DIR


def _get_virtualenv_info() -> tuple[str, Path, Path]:
    global _VIRTUALENV_INFO
    if _VIRTUALENV_INFO is None:
        _VIRTUALENV_INFO = _resolve_virtualenv_info()
    return _VIRTUALENV_INFO


def _start_backend() -> subprocess.Popen[bytes]:
    python_exe, scripts_dir, venv_dir = _get_virtualenv_info()

    cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)

    existing_py_path = env.get("PYTHONPATH")
    if existing_py_path:
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + existing_py_path
    else:
        env["PYTHONPATH"] = str(ROOT_DIR)

    print("[launcher] Starting backend with:", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=ROOT_DIR, env=env)


def _start_frontend() -> subprocess.Popen[bytes]:
    npm_cmd = _command_for("npm")
    cmd = npm_cmd + ["run", "dev", "--", "--host"]

    print("[launcher] Starting frontend with:", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=FRONTEND_DIR)


def _wait_for_weaviate(url: str, timeout: float = 120.0, interval: float = 2.0) -> None:
    print(f"[launcher] Waiting for Weaviate at {url} ...")
    deadline = time.time() + timeout

    while True:
        try:
            with urlopen(url, timeout=5) as response:  # nosec: urllib is used for readiness check
                status = getattr(response, "status", None) or response.getcode()
                if 200 <= status < 300:
                    print(f"[launcher] Weaviate ready (status {status}).")
                    return
                print(
                    f"[launcher] Weaviate responded with status {status}, retrying in {interval} seconds..."
                )
        except URLError as err:
            print(f"[launcher] Weaviate not ready ({err}). Retrying in {interval} seconds...")
        except OSError as err:
            print(f"[launcher] Could not reach Weaviate ({err}). Retrying in {interval} seconds...")

        if time.time() >= deadline:
            raise TimeoutError(f"Timed out after {timeout} seconds waiting for Weaviate at {url}.")

        time.sleep(interval)


def _docker_compose_base_command() -> list[str]:
    docker = shutil.which("docker")
    if docker:
        try:
            result = subprocess.run(
                [docker, "compose", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return [docker, "compose"]
        except OSError:
            pass

    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]

    raise SystemExit(
        "Docker Compose CLI not found. Install Docker Desktop or docker-compose, then retry."
    )


def _ensure_weaviate_container() -> None:
    base_cmd = _docker_compose_base_command()
    full_cmd = base_cmd + ["up", "-d", "weaviate"]

    print("[launcher] Ensuring Weaviate container is running with:", " ".join(full_cmd))
    try:
        subprocess.run(full_cmd, cwd=ROOT_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Failed to start Weaviate via Docker Compose.") from exc
    except FileNotFoundError as exc:
        raise SystemExit("Docker command not found. Make sure Docker Desktop is running.") from exc


def _shutdown(_signum: int, _frame) -> None:  # type: ignore[override]
    print("\n[launcher] Shutting down...")
    for proc in _PROCESSES:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + 5
    for proc in _PROCESSES:
        if proc.poll() is not None:
            continue
        timeout = max(0, deadline - time.time())
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()

    sys.exit(0)


def main() -> None:
    _ensure_directory(BACKEND_DIR, "backend directory")
    _ensure_directory(FRONTEND_DIR, "frontend directory")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    venv_python, _, venv_dir = _get_virtualenv_info()
    print(f"[launcher] Using virtual environment at {venv_dir} (python: {venv_python})")

    _ensure_weaviate_container()

    ready_url = os.environ.get("WEAVIATE_READY_URL", DEFAULT_WEAVIATE_READY_URL)
    wait_timeout = float(os.environ.get("WEAVIATE_WAIT_TIMEOUT", "120"))
    poll_interval = float(os.environ.get("WEAVIATE_POLL_INTERVAL", "2"))

    try:
        _wait_for_weaviate(ready_url, timeout=wait_timeout, interval=poll_interval)
    except TimeoutError as exc:
        raise SystemExit(
            "Weaviate did not become ready in time. Ensure the container can start and retry."
        ) from exc

    try:
        backend_proc = _start_backend()
    except FileNotFoundError as exc:
        raise SystemExit(
            "Failed to start backend. Ensure FastAPI dependencies are installed (pip install -r backend/requirements.txt)."
        ) from exc

    _PROCESSES.append(backend_proc)

    try:
        frontend_proc = _start_frontend()
    except FileNotFoundError as exc:
        backend_proc.terminate()
        raise SystemExit(
            "Failed to start frontend. Ensure Node.js/npm dependencies are installed (npm install inside frontend/)."
        ) from exc

    _PROCESSES.append(frontend_proc)

    print("[launcher] Backend:  http://localhost:8000")
    print("[launcher] Frontend: http://localhost:5173")
    print("[launcher] Press Ctrl+C to stop both services.")

    try:
        while True:
            for proc, name in zip(_PROCESSES, ("backend", "frontend")):
                rc = proc.poll()
                if rc is not None:
                    raise SystemExit(f"[launcher] {name} exited with code {rc}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()
