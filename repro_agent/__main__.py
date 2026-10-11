"""Command line: python -m repro_agent 2106.11959 --focus "XGBoost on California Housing" """
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import ReproAgent
from .llm import make_llm

ICONS = {"goal": "🎯", "step": "─", "thought": "💭", "tool_call": "🔧", "tool_result": "👁 ", "claim": "📌",
         "plan": "🗺 ", "run_start": "▶ ", "run_end": "■ ", "reflect": "🤔", "lesson": "🧠", "install": "📦",
         "nudge": "⚠ ", "verdict": "⚖ ", "error": "✖ ", "paper": "📄", "done": "✔ "}


def load_dotenv(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def print_event(ev: dict):
    t, d = ev["type"], ev["data"]
    icon = ICONS.get(t, "·")
    ts = f"[{ev['t']:6.1f}s]"
    if t == "step":
        print(f"\n{ts} ── step {d['n']} " + "─" * 40)
    elif t == "thought":
        print(f"{ts} {icon} {d['text'][:600]}")
    elif t == "tool_call":
        print(f"{ts} {icon} {d['name']}({', '.join(f'{k}={v!r}' for k, v in d['args'].items())[:300]})")
    elif t == "tool_result":
        status = "ok" if d["ok"] else "ERROR"
        first = d["content"].strip().splitlines()[0][:160] if d["content"].strip() else ""
        print(f"{ts} {icon} {status}: {first}")
    elif t == "claim":
        print(f"{ts} {icon} CLAIM: {d['claim_text']}  [{d['location']}] = {d['reported_value']}{d.get('unit', '')}")
    elif t == "plan":
        mark = {"done": "✓", "doing": "→", "todo": " ", "skipped": "-"}
        print(f"{ts} {icon} PLAN: " + " | ".join(f"{mark[s['status']]} {s['step'][:50]}" for s in d["steps"]))
    elif t == "run_end":
        res = ", ".join(f"{k}={v:g}" for k, v in d["results"].items())
        print(f"{ts} {icon} {'OK' if d['ok'] else 'FAILED'} in {d['seconds']}s {res}")
    elif t == "verdict":
        print("\n" + "═" * 60)
        print(f"  {icon} {d['label']}")
        if d.get("measured") is not None:
            print(f"  paper: {d['reported']}{d['unit']}   ours: {d['measured']:.4g}{d['unit']}   tolerance ±{d['tolerance']:.3g}")
        elif d.get("note"):
            print(f"  {d['note']}")
        print("═" * 60)
    elif t in ("goal", "paper", "reflect", "lesson", "install", "nudge", "error", "done", "run_start"):
        if t == "run_start":
            print(f"{ts} {icon} running: {d['purpose']}")
        else:
            print(f"{ts} {icon} {t.upper()}: " + ", ".join(f"{k}={v}" for k, v in d.items())[:300])


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser(prog="repro", description="Autonomous paper reproduction agent")
    ap.add_argument("source", help="arXiv id / URL, PDF URL, or local PDF/text file")
    ap.add_argument("--focus", default="", help="optional hint: which claim or table to target")
    ap.add_argument("--provider", default=os.getenv("REPRO_PROVIDER", "anthropic"),
                    help="anthropic | openai | gemini | groq | openrouter | ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-runs", type=int, default=14)
    ap.add_argument("--run-timeout", type=int, default=300, help="default seconds per code run")
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--offline", action="store_true", help="skip online dataset search")
    ap.add_argument("--no-install", action="store_true", help="forbid pip installs")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emoji on Windows consoles
    except Exception:
        pass
    try:
        llm = make_llm(a.provider, a.model, base_url=a.base_url)
    except Exception as e:
        print(f"Setup error: {e}\nAdd your key to .env (see .env.example) or pass --provider.")
        return 2
    agent = ReproAgent(llm, runs_dir=a.runs_dir, max_steps=a.max_steps, max_runs=a.max_runs,
                       run_timeout=a.run_timeout, max_minutes=a.max_minutes, online=not a.offline,
                       allow_install=not a.no_install, on_event=print_event)
    out = agent.run(a.source, a.focus)
    print(f"\nReport: {out.report_path}\nHTML:   {out.workdir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
