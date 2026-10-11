"""Offline tests: a scripted fake LLM drives the real loop, tools, sandbox, verdict and report."""
from pathlib import Path

import pytest

from repro_agent.agent import ReproAgent
from repro_agent.llm import AnthropicLLM, AssistantTurn, OpenAICompatLLM, ToolCall
from repro_agent.paper import load_paper, parse_arxiv_id
from repro_agent.verdict import BLOCKED, NOT_REPRODUCED, PARTIAL, REPRODUCED, judge

ROOT = Path(__file__).resolve().parents[1]
TOY = str(ROOT / "examples" / "toy_paper.txt")

BROKEN = "import pandas as pd\nfrom sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)\nprint(X.shape)\nraise ValueError('demo failure: forgot to split')\n"
FIXED = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xtr, ytr)
print(f"RESULT: test_accuracy={m.score(Xte, yte):.4f}")
"""


class ScriptedLLM:
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    def chat(self, system, messages, tools):
        self.calls += 1
        assert "Repro" in system
        if not self.turns:
            return AssistantTurn(text="(script exhausted)")
        text, calls = self.turns.pop(0)
        return AssistantTurn(text=text, tool_calls=[ToolCall(f"c{self.calls}_{i}", n, a) for i, (n, a) in enumerate(calls)])


CLAIM = {"claim_text": "Logistic regression reaches 98.2% test accuracy on WDBC", "location": "Table 1, page 1",
         "method": "Logistic regression (L2, C=1)", "dataset": "WDBC (load_breast_cancer)", "metric": "test accuracy",
         "reported_value": 98.2, "unit": "%", "higher_is_better": True,
         "setup": "stratified 70/30 split, seed 42, standardized"}


def happy_script(measured=98.83):
    return [
        ("Loading the paper.", [("fetch_paper", {"source": TOY})]),
        ("", [("search_paper", {"query": "Table 1 test accuracy logistic regression"}), ("check_environment", {})]),
        ("Logistic regression in Table 1 is feasible.", [("record_claim", CLAIM)]),
        ("", [("update_plan", {"steps": [{"step": "load data", "status": "doing"}, {"step": "train LR", "status": "todo"}]})]),
        ("", [("search_datasets", {"query": "breast cancer"})]),
        ("First attempt.", [("run_python", {"purpose": "smoke test", "code": BROKEN, "timeout_s": 60})]),
        ("Forgot the split; fixing.", [("run_python", {"purpose": "full run", "code": FIXED, "timeout_s": 120})]),
        ("Try to claim a number I never measured.", [("finish", {"outcome": "measured", "measured_value": 99.9, "explanation": "x"})]),
        ("Use the real number.", [("save_lesson", {"lesson": "Always print metrics as RESULT lines"}),
                                  ("finish", {"outcome": "measured", "measured_value": measured, "result_key": "test_accuracy",
                                              "explanation": "Ran LR with the paper's split.", "deviations": ["none"]})]),
    ]


def make_agent(tmp_path, turns, **kw):
    return ReproAgent(ScriptedLLM(turns), runs_dir=tmp_path / "runs", memory_path=tmp_path / "mem.jsonl",
                      online=False, allow_install=False, **kw)


def test_full_loop_reproduces(tmp_path):
    agent = make_agent(tmp_path, happy_script())
    out = agent.run(TOY)
    v = out.verdict
    assert v.label == REPRODUCED, v
    assert v.measured == pytest.approx(98.83)
    assert len(out.state.runs) == 2 and not out.state.runs[0]["ok"] and out.state.runs[1]["ok"]
    assert out.state.runs[1]["results"]["test_accuracy"] == pytest.approx(0.9883)
    kinds = [e["type"] for e in out.trace]
    for k in ("goal", "paper", "claim", "plan", "run_start", "run_end", "lesson", "verdict", "done"):
        assert k in kinds, k
    # the fabricated 99.9 was rejected before the honest finish
    rejected = [e for e in out.trace if e["type"] == "tool_result" and e["data"]["name"] == "finish" and not e["data"]["ok"]]
    assert rejected and "does not appear" in rejected[0]["data"]["content"]
    for f in ("report.md", "report.html", "trace.json", "transcript.json", "comparison.png"):
        assert (out.workdir / f).exists(), f
    assert "REPRODUCED" in (out.workdir / "report.md").read_text()
    assert (tmp_path / "mem.jsonl").read_text().count("RESULT") == 1


def test_lessons_reach_next_run(tmp_path):
    make_agent(tmp_path, happy_script()).run(TOY)
    seen = {}

    class Spy(ScriptedLLM):
        def chat(self, system, messages, tools):
            seen["system"] = system
            return super().chat(system, messages, tools)

    ReproAgent(Spy([("", [("finish", {"outcome": "blocked", "explanation": "x", "blocked_reason": "nothing"})])]),
               runs_dir=tmp_path / "runs", memory_path=tmp_path / "mem.jsonl", online=False).run(TOY)
    assert "Always print metrics as RESULT lines" in seen["system"]


def test_blocked_and_forced_finish(tmp_path):
    turns = [("", [("fetch_paper", {"source": TOY})]),
             ("", [("finish", {"outcome": "blocked", "explanation": "x", "blocked_reason": "short"})]),
             ("", [("finish", {"outcome": "blocked", "explanation": "Needs 8 GPUs.",
                               "blocked_reason": "The claim requires 8xA100 for 3 days of training."})])]
    out = make_agent(tmp_path, turns).run(TOY)
    assert out.verdict.label == BLOCKED and "A100" in out.verdict.note

    # model that never finishes: loop must stop and force a blocked verdict
    out2 = make_agent(tmp_path, [("thinking...", [])] * 10, max_steps=6).run(TOY)
    assert out2.verdict.label == BLOCKED and out2.stopped_reason


def test_run_budget_and_reflection(tmp_path):
    bad = ("", [("run_python", {"purpose": "fail", "code": "raise SystemExit(3)"})])
    turns = [("", [("fetch_paper", {"source": TOY})]), ("", [("record_claim", CLAIM)])] + [bad] * 5
    out = make_agent(tmp_path, turns, max_runs=4, max_steps=9).run(TOY)
    assert len(out.state.runs) == 4
    assert any(e["type"] == "reflect" for e in out.trace)
    assert any("budget" in e["data"].get("content", "") for e in out.trace if e["type"] == "tool_result")


def test_sandbox_scrubs_secrets_and_times_out(tmp_path, monkeypatch):
    from repro_agent.sandbox import Sandbox

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    sb = Sandbox(tmp_path)
    r = sb.run("import os\nprint('KEY' , os.getenv('ANTHROPIC_API_KEY'))\nopen('out.txt','w').write('x')\nprint('RESULT: rmse = 0.45')")
    assert r.ok and "None" in r.stdout and r.results == {"rmse": 0.45} and "out.txt" in r.new_files
    r2 = sb.run("import time\ntime.sleep(5)", timeout=1)
    assert r2.timed_out and not r2.ok


def test_verdict_rules():
    assert judge(98.2, 0.9883, "%", True).label == REPRODUCED  # fraction auto-converted
    assert judge(98.2, 95.0, "%", True).label == NOT_REPRODUCED
    assert judge(98.2, 96.0, "%", True).label == PARTIAL
    assert judge(0.45, 0.43, "", False).label == REPRODUCED  # lower RMSE is better
    assert judge(0.45, 0.60, "", False).label == NOT_REPRODUCED
    assert judge(None, None, "", blocked_reason="gpu").label == BLOCKED


def test_paper_parsing():
    p = load_paper(TOY)
    assert "Table 1" in p.overview()
    assert "98.2" in p.search("logistic regression accuracy Table 1")
    assert parse_arxiv_id("https://arxiv.org/abs/2106.11959v3") == "2106.11959v3"
    assert parse_arxiv_id("2106.11959") == "2106.11959"
    assert parse_arxiv_id("arXiv:1412.6980") == "1412.6980"
    assert parse_arxiv_id("hello world") is None


def test_adapter_wire_formats():
    msgs = [{"role": "user", "text": "go"},
            {"role": "assistant", "text": "hi", "tool_calls": [{"id": "t1", "name": "check_environment", "args": {}}]},
            {"role": "tool", "results": [{"id": "t1", "name": "check_environment", "content": "ok", "is_error": False}]},
            {"role": "user", "text": "reflect"}]
    a = AnthropicLLM.to_wire(msgs)
    assert [m["role"] for m in a] == ["user", "assistant", "user"]  # tool result + reflect merged
    assert a[2]["content"][0]["type"] == "tool_result" and a[2]["content"][1]["type"] == "text"
    o = OpenAICompatLLM.to_wire("sys", msgs)
    assert [m["role"] for m in o] == ["system", "user", "assistant", "tool", "user"]
    assert o[2]["tool_calls"][0]["function"]["arguments"] == "{}"
