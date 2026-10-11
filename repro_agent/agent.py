"""The agent loop: goal -> decide -> use tool -> observe -> (adapt) -> finish."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .llm import LLMError
from .memory import LessonMemory
from .sandbox import Sandbox
from .tools import TOOL_SPECS, RunState, ToolBox
from .verdict import Verdict, judge

SYSTEM_PROMPT = """You are Repro, an autonomous research engineer. Your job: take a machine-learning paper, pick ONE quantitative claim, re-run the experiment yourself, and report honestly whether the number holds up.

Work like this:
1. fetch_paper, then check_environment. Skim the overview, then use search_paper to find results tables and experimental details.
2. Choose ONE claim that is (a) a specific number in the paper, (b) central or representative, and (c) feasible on THIS machine in about {minutes} minutes of compute: prefer classical/baseline models, public datasets, CPU-friendly sizes. If the user gave a focus hint, follow it when feasible. Call record_claim with the exact reported number and its location.
3. Call update_plan with 4-7 concrete steps, and update statuses as you progress.
4. Find the data (search_datasets), write a self-contained script, and run it with run_python. Follow the paper's split, preprocessing and hyperparameters where stated; where the paper is silent, use standard defaults and note each one as a deviation. Use fixed seeds. Debug on a subsample first, then run the full setting. Print final metrics as `RESULT: key=value`.
5. When a run fails, read the error, find the root cause, and change the approach. Never resubmit the same failing code. If a missing package is the cause, use install_package. After solving a non-obvious problem, save_lesson with a short general lesson.
6. If time allows, run a quick robustness check (e.g. 2-3 seeds) and report the mean.
7. Call finish. The measured_value MUST be a number your code printed. List every deviation from the paper.
If no claim is feasible (needs GPUs, private data, or missing critical details), finish with outcome='blocked' and a precise reason. An honest "could not attempt" beats a fake result.

Pacing: spend at most a third of your step budget reading the paper and code. A faithful-enough experiment with documented deviations beats a perfect setup that never runs. Record the claim early and get a first result before refining.

Rules: never invent or estimate numbers; keep each run under {run_timeout}s unless necessary; do not download more than ~1 GB; keep your messages short, the tools do the work. Budget: {max_steps} steps, {max_runs} code runs.

Lessons from earlier runs (apply them):
{lessons}
"""

REFLECT_PROMPT = ("Three code runs in a row have failed. Stop and reflect before the next action: "
                  "state the root cause in one sentence, decide whether to simplify (smaller data, simpler model, "
                  "different loader) or switch claim, then continue.")
NUDGE_PROMPT = "Continue by calling a tool. If you are done, call finish."


@dataclass
class RunOutcome:
    run_id: str
    workdir: Path
    verdict: Verdict
    state: RunState
    trace: list[dict] = field(default_factory=list)
    report_path: Path | None = None
    stopped_reason: str = ""


def _compact(messages: list[dict], keep_last: int = 8, max_old: int = 700) -> list[dict]:
    """Trim old tool outputs so long runs stay inside the context window."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool_idx[:-keep_last])
    out = []
    for i, m in enumerate(messages):
        if i in old:
            m = {"role": "tool", "results": [
                {**r, "content": r["content"] if len(r["content"]) <= max_old else
                 r["content"][:max_old] + "\n...[older output trimmed]"} for r in m["results"]]}
        out.append(m)
    return out


class ReproAgent:
    def __init__(self, llm, runs_dir: str | Path = "runs", memory_path: str | Path = "memory/lessons.jsonl",
                 max_steps: int = 40, max_runs: int = 14, run_timeout: int = 300, max_minutes: float = 30,
                 online: bool = True, allow_install: bool = True, on_event: Callable[[dict], None] | None = None):
        self.llm = llm
        self.runs_dir = Path(runs_dir)
        self.memory = LessonMemory(Path(memory_path))
        self.max_steps, self.max_runs, self.run_timeout = max_steps, max_runs, run_timeout
        self.max_minutes, self.online, self.allow_install = max_minutes, online, allow_install
        self.on_event = on_event
        self.trace: list[dict] = []
        self.t0 = time.time()

    def emit(self, kind: str, data: dict | None = None):
        ev = {"t": round(time.time() - self.t0, 2), "type": kind, "data": data or {}}
        self.trace.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass  # a UI bug must not kill the run

    def run(self, source: str, focus: str = "") -> RunOutcome:
        self.t0 = time.time()
        self.trace = []
        slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(source).stem if Path(source).exists() else source)[-40:].strip("-")
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + (slug or "paper")
        workdir = self.runs_dir / run_id
        state = RunState(workdir=workdir, sandbox=Sandbox(workdir / "experiment"), memory=self.memory,
                         online=self.online, allow_install=self.allow_install)
        tools = ToolBox(state, self.emit)
        system = SYSTEM_PROMPT.format(minutes=int(self.max_minutes * 0.6), run_timeout=self.run_timeout,
                                      max_steps=self.max_steps, max_runs=self.max_runs,
                                      lessons=self.memory.as_prompt())
        goal = f"Reproduce one quantitative claim from this paper: {source}"
        if focus.strip():
            goal += f"\nFocus hint from the user: {focus.strip()}"
        messages: list[dict] = [{"role": "user", "text": goal}]
        self.emit("goal", {"source": source, "focus": focus, "run_id": run_id,
                           "model": getattr(self.llm, "model", "?")})

        stopped, nudges, llm_down = "", 0, False
        nudged: set[str] = set()
        for step in range(1, self.max_steps + 1):
            if (time.time() - self.t0) / 60 > self.max_minutes:
                stopped = "time budget exhausted"
                break
            self.emit("step", {"n": step})
            try:
                turn = self.llm.chat(system, _compact(messages), TOOL_SPECS)
            except LLMError as e:  # keep the work done so far: write a partial report instead of crashing
                self.emit("error", {"text": f"LLM unavailable: {e}"})
                stopped, llm_down = f"LLM unavailable ({str(e)[:160]})", True
                break
            calls = [{"id": tc.id, "name": tc.name, "args": tc.args, "extra": tc.extra} for tc in turn.tool_calls]
            messages.append({"role": "assistant", "text": turn.text, "tool_calls": calls, "extra": turn.extra})
            if turn.text:
                self.emit("thought", {"text": turn.text})
            if not calls:
                nudges += 1
                if nudges > 3:
                    stopped = "agent stopped calling tools"
                    break
                messages.append({"role": "user", "text": NUDGE_PROMPT})
                continue
            nudges = 0
            results = []
            for c in calls:
                if c["name"] == "run_python" and len(state.runs) >= self.max_runs:
                    content, is_err = f"Run budget ({self.max_runs}) exhausted. Call finish now.", True
                else:
                    if c["name"] == "run_python":
                        c["args"].setdefault("timeout_s", self.run_timeout)
                    self.emit("tool_call", {"name": c["name"], "args": _brief_args(c["name"], c["args"])})
                    content, is_err = tools.dispatch(c["name"], c["args"])
                self.emit("tool_result", {"name": c["name"], "ok": not is_err, "content": content[:4000]})
                results.append({"id": c["id"], "name": c["name"], "content": content, "is_error": is_err})
                if state.finished:
                    break
            # every assistant tool call needs a result, even ones skipped after finish
            done_ids = {r["id"] for r in results}
            for c in calls:
                if c["id"] not in done_ids:
                    results.append({"id": c["id"], "name": c["name"], "content": "Skipped: run already finished.", "is_error": True})
            results[-1]["content"] += f"\n[budget: step {step}/{self.max_steps}, runs {len(state.runs)}/{self.max_runs}, {int(time.time() - self.t0)}s elapsed]"
            messages.append({"role": "tool", "results": results})
            if state.finished:
                break
            self._pace(step, state, messages, nudged)
            if state.fail_streak and state.fail_streak % 3 == 0 and calls[-1]["name"] == "run_python":
                self.emit("reflect", {"fail_streak": state.fail_streak})
                messages.append({"role": "user", "text": REFLECT_PROMPT})
        else:
            stopped = "step budget exhausted"

        if not state.finished and not llm_down:
            self._force_finish(system, messages, tools, state, stopped)
        if not state.finished:
            state.finished = {"outcome": "blocked", "blocked_reason": f"Run interrupted: {stopped}",
                              "explanation": "The run stopped before a verified result. Partial work is listed below.",
                              "deviations": []}

        verdict = self._verdict(state)
        self.emit("verdict", verdict.to_dict())
        outcome = RunOutcome(run_id, workdir, verdict, state, self.trace, stopped_reason=stopped)
        from .report import write_outputs

        outcome.report_path = write_outputs(outcome, messages)
        self.emit("done", {"report": str(outcome.report_path)})
        return outcome

    def _pace(self, step: int, state: RunState, messages: list[dict], nudged: set[str]):
        """Budget-aware self-monitoring: stop over-researching and get to an experiment."""
        left = self.max_steps - step
        if not state.claim and step >= max(3, int(self.max_steps * 0.35)) and "claim" not in nudged:
            nudged.add("claim")
            text = (f"Pacing check: {step} of {self.max_steps} steps used and no claim recorded yet. "
                    "Stop researching. Call record_claim now with the best-supported number, then run an experiment.")
        elif not state.successful_runs_with_results() and step >= max(5, int(self.max_steps * 0.55)) and "run" not in nudged:
            nudged.add("run")
            text = (f"Pacing check: only {left} steps left and no measured result yet. Your next action must be "
                    "run_python with a complete, simple experiment that prints a RESULT line. List approximations as deviations.")
        elif left == 2 and "finish" not in nudged:
            nudged.add("finish")
            text = "Two steps left. If you have a measured result, call finish now."
        else:
            return
        self.emit("nudge", {"text": text})
        messages.append({"role": "user", "text": text})

    def _force_finish(self, system, messages, tools, state, why):
        self.emit("nudge", {"text": f"Forcing finish: {why or 'no finish call'}"})
        messages.append({"role": "user", "text": f"Stop now ({why}). Call finish with your best supported result, "
                                                 "or outcome='blocked' if you have none."})
        try:  # full tool list keeps provider history valid; only finish is honoured
            turn = self.llm.chat(system, _compact(messages), TOOL_SPECS)
            for tc in turn.tool_calls:
                if tc.name == "finish":
                    content, _ = tools.dispatch("finish", tc.args)
                    self.emit("tool_result", {"name": "finish", "ok": bool(state.finished), "content": content})
                    break
        except Exception as e:
            self.emit("error", {"text": f"final finish call failed: {e}"})
        if not state.finished:
            state.finished = {"outcome": "blocked", "blocked_reason": f"Agent did not finish: {why or 'unknown'}",
                              "explanation": "The run ended before a verified result.", "deviations": []}

    @staticmethod
    def _verdict(state: RunState) -> Verdict:
        f, c = state.finished, state.claim
        if f["outcome"] == "blocked" or not c:
            return judge(c["reported_value"] if c else None, None, (c or {}).get("unit", ""),
                         blocked_reason=f.get("blocked_reason") or "no claim recorded")
        return judge(float(c["reported_value"]), float(f["measured_value"]), c.get("unit", ""),
                     c.get("higher_is_better"))


def _brief_args(name: str, args: dict) -> dict:
    if name == "run_python":
        return {"purpose": args.get("purpose", ""), "lines": len(str(args.get("code", "")).splitlines())}
    return {k: (v if len(json.dumps(v, default=str)) < 400 else str(v)[:400] + "...") for k, v in args.items()}
