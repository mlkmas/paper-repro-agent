"""Run agent-written Python in a separate process with a timeout and a scrubbed environment.

This is isolation for robustness (crashes, infinite loops, leaked API keys), not a
security boundary. For untrusted papers run the whole app inside Docker (see README).
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
RESULT_LINE = re.compile(r"^RESULT:\s*([A-Za-z0-9_\-\.\s/@%]+?)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M)


@dataclass
class RunResult:
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool
    script: str
    new_files: list[str]
    results: dict[str, float]

    def summary(self, max_chars: int = 6000) -> str:
        status = "TIMEOUT" if self.timed_out else ("OK" if self.ok else f"FAILED (exit {self.returncode})")
        parts = [f"status: {status}   time: {self.seconds:.1f}s   script: {self.script}"]
        if self.results:
            parts.append("parsed RESULT lines: " + ", ".join(f"{k}={v:g}" for k, v in self.results.items()))
        if self.new_files:
            parts.append("new files: " + ", ".join(self.new_files[:20]))
        parts.append("stdout:\n" + clip(self.stdout, max_chars * 2 // 3))
        if self.stderr.strip():
            parts.append("stderr:\n" + clip(self.stderr, max_chars // 3))
        return "\n".join(parts)


def clip(text: str, n: int) -> str:
    """Keep head and tail: errors live at the end, setup info at the start."""
    text = text or ""
    if len(text) <= n:
        return text
    head = n // 3
    return text[:head] + f"\n...[{len(text) - n} chars omitted]...\n" + text[-(n - head):]


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in SECRET_HINTS)}
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _limits(mem_mb: int):
    def apply():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        except Exception:
            pass
    return apply


class Sandbox:
    def __init__(self, workdir: Path, mem_mb: int = 0):
        """mem_mb > 0 caps address space on Linux (off by default: torch reserves huge virtual memory)."""
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.mem_mb = mem_mb
        self.counter = 0

    def _snapshot(self) -> dict[str, float]:
        return {str(p.relative_to(self.workdir)): p.stat().st_mtime
                for p in self.workdir.rglob("*") if p.is_file() and "__pycache__" not in p.parts}

    def run(self, code: str, timeout: int = 300) -> RunResult:
        self.counter += 1
        script = self.workdir / f"step_{self.counter:02d}.py"
        script.write_text(code, encoding="utf-8")
        before = self._snapshot()
        start = time.time()
        kwargs = dict(cwd=self.workdir, env=clean_env(), capture_output=True, text=True,
                      encoding="utf-8", errors="replace", timeout=timeout)
        if platform.system() == "Linux" and self.mem_mb > 0:
            kwargs["preexec_fn"] = _limits(self.mem_mb)
        try:
            p = subprocess.run([sys.executable, "-u", script.name], **kwargs)
            timed_out, rc, out, err = False, p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            timed_out, rc = True, None
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + \
                f"\nKilled after {timeout}s timeout. Reduce data size/epochs or raise timeout_s."
        after = self._snapshot()
        new = sorted(k for k, t in after.items() if (k not in before or before[k] != t) and k != script.name)
        results = {m.group(1).strip(): float(m.group(2)) for m in RESULT_LINE.finditer(out)}
        return RunResult(ok=(rc == 0 and not timed_out), returncode=rc, stdout=out, stderr=err,
                         seconds=time.time() - start, timed_out=timed_out, script=script.name,
                         new_files=new, results=results)


ALLOWED_PACKAGES = {
    "xgboost", "lightgbm", "catboost", "torch", "torchvision", "torchaudio", "datasets", "openml",
    "ucimlrepo", "imbalanced-learn", "statsmodels", "scipy", "matplotlib", "seaborn", "tqdm",
    "transformers", "sentence-transformers", "tabulate", "optuna", "networkx", "gensim", "nltk",
    "scikit-image", "pillow", "pyarrow", "polars", "einops", "timm", "gymnasium", "shap",
}


def pip_install(package: str, timeout: int = 600) -> tuple[bool, str]:
    name = re.split(r"[<>=!\[ ]", package.strip(), maxsplit=1)[0].lower()
    if name not in ALLOWED_PACKAGES:
        return False, f"'{name}' is not on the install allowlist: {', '.join(sorted(ALLOWED_PACKAGES))}"
    try:
        p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "pip install timed out"
    msg = clip((p.stdout or "") + (p.stderr or ""), 1500) or "installed"
    return p.returncode == 0, msg


def environment_report() -> str:
    import importlib.util

    lines = [f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}",
             f"cpu cores: {os.cpu_count()}"]
    try:
        import psutil  # optional

        lines.append(f"ram: {psutil.virtual_memory().total / 2**30:.1f} GB")
    except Exception:
        pass
    pkgs = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "xgboost", "lightgbm", "torch",
            "torchvision", "datasets", "transformers", "imblearn", "statsmodels"]
    have = [p for p in pkgs if importlib.util.find_spec(p)]
    lines.append("installed: " + ", ".join(have))
    lines.append("missing (installable via install_package if allowlisted): " + ", ".join(p for p in pkgs if p not in have))
    gpu = "none detected"
    if importlib.util.find_spec("torch"):
        try:
            import torch

            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                gpu = "Apple MPS"
        except Exception:
            pass
    lines.append(f"gpu: {gpu}")
    return "\n".join(lines)
