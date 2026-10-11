"""The agent's tools: JSON schemas plus the code that executes them against shared run state."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import datasets as ds
from .memory import LessonMemory
from .paper import Paper, load_paper
from .sandbox import Sandbox, environment_report, pip_install

PLAN_STATUSES = ["todo", "doing", "done", "skipped"]

TOOL_SPECS: list[dict] = [
    {
        "name": "fetch_paper",
        "description": "Download and parse the paper. Accepts an arXiv id (e.g. 2106.11959), arXiv/PDF URL, or local file path. Returns title, abstract, section headings and table captions.",
        "parameters": {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]},
    },
    {
        "name": "search_paper",
        "description": "Find passages in the loaded paper by keywords (e.g. 'Table 2 accuracy logistic regression', 'learning rate batch size epochs'), or read a whole page with page=N. Use this to find exact reported numbers and experimental details.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords to search for"},
            "page": {"type": "integer", "description": "1-based page number to read in full"}}},
    },
    {
        "name": "check_environment",
        "description": "Report Python version, CPU cores, GPU and installed ML packages. Use before choosing a claim so you pick something feasible here.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "record_claim",
        "description": "Commit to ONE quantitative claim from the paper to reproduce. Call again only if you must switch claims (say why).",
        "parameters": {"type": "object", "properties": {
            "claim_text": {"type": "string", "description": "The claim in one sentence, faithful to the paper"},
            "location": {"type": "string", "description": "Where in the paper, e.g. 'Table 3, page 7'"},
            "method": {"type": "string"},
            "dataset": {"type": "string"},
            "metric": {"type": "string", "description": "e.g. test accuracy, RMSE, AUC"},
            "reported_value": {"type": "number"},
            "unit": {"type": "string", "description": "'%' if the paper reports percent, else '' (fraction) or the unit name"},
            "higher_is_better": {"type": "boolean"},
            "setup": {"type": "string", "description": "Split, preprocessing, hyperparameters, seeds as stated in the paper"},
            "why_this_claim": {"type": "string", "description": "Why it is central and feasible on this machine"}},
            "required": ["claim_text", "location", "method", "dataset", "metric", "reported_value", "unit", "higher_is_better", "setup"]},
    },
    {
        "name": "update_plan",
        "description": "Write or update your step-by-step plan. Send the full list each time with statuses.",
        "parameters": {"type": "object", "properties": {"steps": {"type": "array", "items": {
            "type": "object", "properties": {
                "step": {"type": "string"}, "status": {"type": "string", "enum": PLAN_STATUSES}},
            "required": ["step", "status"]}}}, "required": ["steps"]},
    },
    {
        "name": "search_datasets",
        "description": "Search a built-in catalog plus OpenML and Hugging Face for a dataset. Returns names and the exact loader call.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "run_python",
        "description": (
            "Run a standalone Python script in the experiment folder (files persist between runs). "
            "Print final metrics as lines like `RESULT: test_accuracy=0.9312` so they are captured. "
            "Use fixed random seeds. Debug on a subsample first, then run the full experiment."),
        "parameters": {"type": "object", "properties": {
            "purpose": {"type": "string", "description": "One line: what this run checks"},
            "code": {"type": "string"},
            "timeout_s": {"type": "integer", "description": "Seconds before the run is killed (default 300, max 1200)"}},
            "required": ["purpose", "code"]},
    },
    {
        "name": "install_package",
        "description": "pip install an allowlisted package (e.g. xgboost, lightgbm, torch, datasets, imbalanced-learn) when an import fails.",
        "parameters": {"type": "object", "properties": {"package": {"type": "string"}}, "required": ["package"]},
    },
    {
        "name": "save_lesson",
        "description": "Save a short, general lesson learned from a failure (e.g. a library quirk) so future runs avoid it. Not run-specific facts.",
        "parameters": {"type": "object", "properties": {"lesson": {"type": "string"}}, "required": ["lesson"]},
    },
    {
        "name": "finish",
        "description": (
            "End the run. outcome='measured' requires a recorded claim and a measured value that appears in a "
            "successful run's output. outcome='blocked' if the claim cannot be attempted here (GPU-only, private data, "
            "missing details); explain precisely what is missing."),
        "parameters": {"type": "object", "properties": {
            "outcome": {"type": "string", "enum": ["measured", "blocked"]},
            "measured_value": {"type": "number", "description": "Your final measured value, in the same unit as reported_value"},
            "result_key": {"type": "string", "description": "The RESULT key that holds it, e.g. test_accuracy"},
            "explanation": {"type": "string", "description": "2-5 sentences: what you ran and how to read the result"},
            "deviations": {"type": "array", "items": {"type": "string"}, "description": "Every way your setup differs from the paper's"},
            "blocked_reason": {"type": "string"}},
            "required": ["outcome", "explanation"]},
    },
]


@dataclass
class RunState:
    workdir: Path
    sandbox: Sandbox
    memory: LessonMemory
    online: bool = True
    allow_install: bool = True
    paper: Paper | None = None
    claim: dict | None = None
    claim_history: list[dict] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    installs: list[str] = field(default_factory=list)
    lessons_saved: list[str] = field(default_factory=list)
    finished: dict | None = None
    fail_streak: int = 0

    @property
    def successful_runs(self) -> list[dict]:
        return [r for r in self.runs if r["ok"]]

    def successful_runs_with_results(self) -> list[dict]:
        return [r for r in self.runs if r["ok"] and r["results"]]


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-9)


def _value_in_outputs(state: RunState, value: float, key: str | None) -> tuple[bool, float | None]:
    """Anti-hallucination check: the claimed number must come from a real run."""
    for r in reversed(state.successful_runs):
        if key and key in r["results"]:
            if _close(r["results"][key], value) or _close(r["results"][key] * 100, value):
                return True, r["results"][key]
        for v in r["results"].values():
            if _close(v, value) or _close(v * 100, value):
                return True, v
        for tok in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", r["stdout"]):
            try:
                f = float(tok)
            except ValueError:
                continue
            if _close(f, value) or _close(f * 100, value):
                return True, f
    return False, None


class ToolBox:
    def __init__(self, state: RunState, emit: Callable[[str, dict], None]):
        self.s = state
        self.emit = emit

    def dispatch(self, name: str, args: dict) -> tuple[str, bool]:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return f"Unknown tool '{name}'. Available: {', '.join(t['name'] for t in TOOL_SPECS)}", True
        if "__unparsed__" in args:
            return "Your tool arguments were not valid JSON. Resend them as a JSON object.", True
        try:
            return fn(**args)
        except TypeError as e:
            return f"Bad arguments for {name}: {e}", True
        except Exception as e:  # tools must never crash the loop
            return f"{type(e).__name__}: {e}", True

    # --- paper ---------------------------------------------------------
    def t_fetch_paper(self, source: str):
        self.s.paper = load_paper(source)
        if len(self.s.paper.text) < 500:
            return ("Paper loaded but almost no text was extracted (scanned PDF?). "
                    "Overview:\n" + self.s.paper.overview()), True
        self.emit("paper", {"title": self.s.paper.title, "source": self.s.paper.source, "pages": len(self.s.paper.pages)})
        return self.s.paper.overview(), False

    def t_search_paper(self, query: str = "", page: int | None = None):
        if not self.s.paper:
            return "No paper loaded. Call fetch_paper first.", True
        if page:
            return self.s.paper.page(int(page)), False
        if not query:
            return "Give a query or a page number.", True
        return self.s.paper.search(query), False

    def t_check_environment(self):
        return environment_report(), False

    # --- planning ------------------------------------------------------
    def t_record_claim(self, **claim):
        if not self.s.paper:
            return "Fetch the paper before recording a claim.", True
        required = TOOL_SPECS[3]["parameters"]["required"]
        missing = [k for k in required if claim.get(k) in (None, "") and k != "unit"]
        if missing:
            return f"Missing fields: {', '.join(missing)}", True
        try:
            claim["reported_value"] = float(claim["reported_value"])
        except (TypeError, ValueError):
            return "reported_value must be a number.", True
        claim.setdefault("unit", "")
        if self.s.claim:
            self.s.claim_history.append(self.s.claim)
        self.s.claim = claim
        self.emit("claim", claim)
        switched = " (switched from previous claim)" if self.s.claim_history else ""
        return f"Claim recorded{switched}: {claim['metric']} = {claim['reported_value']}{claim.get('unit', '')} for {claim['method']} on {claim['dataset']}.", False

    def t_update_plan(self, steps: list):
        clean = []
        for st in steps:
            if isinstance(st, dict) and st.get("step"):
                status = st.get("status", "todo")
                clean.append({"step": str(st["step"])[:200], "status": status if status in PLAN_STATUSES else "todo"})
        if not clean:
            return "Plan must be a non-empty list of {step, status}.", True
        self.s.plan = clean
        self.emit("plan", {"steps": clean})
        done = sum(1 for p in clean if p["status"] == "done")
        return f"Plan saved ({done}/{len(clean)} done).", False

    # --- data & code ---------------------------------------------------
    def t_search_datasets(self, query: str):
        return ds.search_datasets(query, online=self.s.online), False

    def t_run_python(self, code: str, purpose: str = "", timeout_s: int = 300):
        timeout_s = max(10, min(int(timeout_s or 300), 1200))
        self.emit("run_start", {"purpose": purpose, "code": code, "timeout_s": timeout_s})
        res = self.s.sandbox.run(code, timeout=timeout_s)
        record = {"purpose": purpose, "ok": res.ok, "script": res.script, "seconds": round(res.seconds, 2),
                  "results": res.results, "stdout": res.stdout[-20000:], "stderr": res.stderr[-8000:],
                  "timed_out": res.timed_out, "new_files": res.new_files}
        self.s.runs.append(record)
        self.s.fail_streak = 0 if res.ok else self.s.fail_streak + 1
        self.emit("run_end", record)
        text = res.summary()
        if res.ok and not res.results:
            text += "\nNote: no `RESULT: key=value` lines were printed; print final metrics that way."
        return text, not res.ok

    def t_install_package(self, package: str):
        if not self.s.allow_install:
            return "Package installs are disabled for this run. Use what is installed.", True
        ok, msg = pip_install(package)
        if ok:
            self.s.installs.append(package)
        self.emit("install", {"package": package, "ok": ok})
        return msg, not ok

    def t_save_lesson(self, lesson: str):
        msg = self.s.memory.add(lesson, source=self.s.paper.source if self.s.paper else "")
        if msg.startswith("Saved"):
            self.s.lessons_saved.append(lesson)
            self.emit("lesson", {"lesson": lesson})
        return msg, False

    # --- end -----------------------------------------------------------
    def t_finish(self, outcome: str, explanation: str, measured_value: float | None = None,
                 result_key: str | None = None, deviations: list | None = None, blocked_reason: str | None = None):
        if outcome == "blocked":
            if not self.s.paper:
                return "Fetch and read the paper before declaring it blocked.", True
            if not blocked_reason or len(blocked_reason) < 15:
                return "Give a specific blocked_reason (what resource or detail is missing).", True
            self.s.finished = {"outcome": "blocked", "blocked_reason": blocked_reason, "explanation": explanation,
                               "deviations": deviations or []}
            return "Run finished as blocked.", False
        if outcome != "measured":
            return "outcome must be 'measured' or 'blocked'.", True
        if not self.s.claim:
            return "Record the claim with record_claim before finishing.", True
        if not self.s.successful_runs:
            return "No successful run yet. Run the experiment, or finish with outcome='blocked'.", True
        if measured_value is None and result_key:
            for r in reversed(self.s.successful_runs):
                if result_key in r["results"]:
                    measured_value = r["results"][result_key]
                    break
        if measured_value is None:
            return "Provide measured_value (or a result_key printed by a successful run).", True
        found, raw = _value_in_outputs(self.s, float(measured_value), result_key)
        if not found:
            return (f"measured_value={measured_value} does not appear in any successful run's output. "
                    "Report only numbers your code actually printed (as a RESULT line)."), True
        self.s.finished = {"outcome": "measured", "measured_value": float(measured_value), "raw_value": raw,
                           "result_key": result_key, "explanation": explanation, "deviations": deviations or []}
        return "Run finished.", False
