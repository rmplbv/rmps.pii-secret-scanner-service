"""Start/stop a llama.cpp server for one GGUF model (binary or docker), or use
an already running one (external)."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx

from .core import peak_rss_gb


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_gguf(repo: str, filename: str, models_dir: Path, offline: bool = False) -> Path:
    path = models_dir / filename
    if path.exists():
        return path
    if offline:
        raise FileNotFoundError(f"{path} missing (offline mode)")
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo}/{filename} ...", flush=True)
    got = hf_hub_download(repo, filename, local_dir=str(models_dir))
    return Path(got)


class LlamaServer:
    def __init__(self, mode: str, model_path: Path | None, *, binary: str | None = None, image: str | None = None,
                 url: str | None = None, ctx_size: int = 8192, threads: int = 6, extra_args: list[str] | None = None,
                 log_path: Path | None = None):
        self.mode = mode
        self.model_path = model_path
        self.binary = binary
        self.image = image or "ghcr.io/ggml-org/llama.cpp:server"
        self.ctx_size, self.threads = ctx_size, threads
        self.extra_args = extra_args or []
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.container: str | None = None
        self.port = free_port()
        self.url = url or f"http://127.0.0.1:{self.port}"

    def _args(self, model: str) -> list[str]:
        return ["-m", model, "--host", "127.0.0.1" if self.mode == "binary" else "0.0.0.0",
                "--port", str(self.port if self.mode == "binary" else 8080),
                "--ctx-size", str(self.ctx_size), "--threads", str(self.threads), "--parallel", "1",
                "--jinja", *self.extra_args]

    def __enter__(self) -> "LlamaServer":
        if self.mode == "external":
            self._wait()
            return self
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        if self.mode == "binary":
            binary = self.binary or os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
            if not binary:
                raise RuntimeError("llama-server binary not found: pass --llama-server or set LLAMA_SERVER")
            env = {**os.environ, "LD_LIBRARY_PATH": f"{Path(binary).parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
            self.proc = subprocess.Popen([binary, *self._args(str(self.model_path))], stdout=log,
                                         stderr=subprocess.STDOUT, env=env)
        elif self.mode == "docker":
            name = f"pii-bench-llm-{self.port}"
            cmd = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{self.port}:8080",
                   "-v", f"{self.model_path.parent.resolve()}:/models:ro", self.image,
                   *self._args(f"/models/{self.model_path.name}")]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            self.container = name
        else:
            raise ValueError(f"unknown llm runner {self.mode!r}")
        self._wait()
        return self

    def _wait(self, timeout: float = 900) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} (see {self.log_path})")
            try:
                if httpx.get(self.url + "/health", timeout=3).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError("llama-server did not become healthy")

    def peak_rss_gb(self) -> float | None:
        if self.proc:
            return peak_rss_gb(self.proc.pid)
        if self.container:
            out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container],
                                 capture_output=True, text=True).stdout
            m = re.match(r"([\d.]+)\s*([KMG]i?B)", out.strip())
            if m:
                f = {"K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1}[m.group(2)[0]]
                return round(float(m.group(1)) * f, 2)
        return None

    def throughput(self) -> dict:
        """Average prompt/generation tokens per second from the server log."""
        if not self.log_path or not Path(self.log_path).exists():
            return {}
        log = Path(self.log_path).read_text(errors="replace")
        pp = [float(x) for x in re.findall(r"prompt eval time =.*?([\d.]+) tokens per second", log)]
        tg = [float(x) for x in re.findall(r"\s eval time =.*?([\d.]+) tokens per second", log)]
        out_tok = sum(int(x) for x in re.findall(r"\s eval time =\s*[\d.]+ ms /\s*(\d+) tokens", log))
        return {"prompt_tok_s": round(sum(pp) / len(pp), 1) if pp else None,
                "gen_tok_s": round(sum(tg) / len(tg), 2) if tg else None, "generated_tokens": out_tok}

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.container:
            subprocess.run(["docker", "stop", self.container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
