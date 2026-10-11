"""Streamlit UI: watch the agent loop live, or replay a saved run (for demo videos).

    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from repro_agent.__main__ import load_dotenv
from repro_agent.agent import ReproAgent
from repro_agent.llm import OPENAI_COMPAT_PRESETS, make_llm
from repro_agent.memory import LessonMemory

load_dotenv()
st.set_page_config(page_title="Repro: paper reproduction agent", page_icon="⚖️", layout="wide")

RUNS = Path("runs")
MEMORY = Path("memory/lessons.jsonl")
VERDICT_COLORS = {"REPRODUCED": "#1f8a4c", "PARTIALLY REPRODUCED": "#b7791f",
                  "NOT REPRODUCED": "#c0392b", "COULD NOT ATTEMPT": "#5f6b7a"}
PHASES = ["Goal", "Decide", "Use tool", "Observe", "Finish"]
PHASE_OF = {"goal": 0, "thought": 1, "step": 1, "tool_call": 2, "run_start": 2, "tool_result": 3,
            "run_end": 3, "reflect": 1, "verdict": 4, "done": 4}
TOOL_ICONS = {"fetch_paper": "📄", "search_paper": "🔎", "check_environment": "🖥️", "record_claim": "📌",
              "update_plan": "🗺️", "search_datasets": "🗃️", "run_python": "🐍", "install_package": "📦",
              "save_lesson": "🧠", "finish": "🏁"}

st.markdown("""<style>
.loop{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 14px}
.loop span{padding:5px 12px;border-radius:999px;border:1px solid #c9d1da;font-size:13px;color:#5f6b7a}
.loop span.on{background:#1d2330;color:#fff;border-color:#1d2330}
.card{border:1px solid #dfe4ea;border-radius:10px;padding:12px 14px;margin-bottom:10px;font-size:14px}
.card h4{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#5f6b7a}
.verdict{border-radius:12px;padding:16px 18px;color:#fff;margin:8px 0}
.verdict .big{font-size:26px;font-weight:700}
.plan div{margin:2px 0}
@media (prefers-color-scheme: dark){.card{border-color:#3a4250}.card h4{color:#9aa5b1}.loop span{border-color:#3a4250;color:#9aa5b1}.loop span.on{background:#e8ecf1;color:#111}}
</style>""", unsafe_allow_html=True)


class TraceView:
    """Renders agent events. Used identically for live runs and replays."""

    def __init__(self):
        self.loop_ph = st.empty()
        left, right = st.columns([3, 2], gap="large")
        with right:
            self.paper_ph, self.claim_ph, self.plan_ph, self.stats_ph = st.empty(), st.empty(), st.empty(), st.empty()
        self.left = left
        self.verdict_ph = right.empty()
        self.status = None
        self.runs = self.fails = self.steps = 0
        self.lessons: list[str] = []
        self.render_loop(0)

    def render_loop(self, idx: int):
        chips = "".join(f'<span class="{"on" if i == idx else ""}">{p}</span>' for i, p in enumerate(PHASES))
        self.loop_ph.markdown(f'<div class="loop">{chips}</div>', unsafe_allow_html=True)

    def stats(self):
        extra = "".join(f"<div>🧠 {l}</div>" for l in self.lessons)
        self.stats_ph.markdown(
            f'<div class="card"><h4>Progress</h4>steps {self.steps} · code runs {self.runs} · failed runs recovered from: {self.fails}{extra}</div>',
            unsafe_allow_html=True)

    def handle(self, ev: dict):
        t, d = ev["type"], ev["data"]
        if t in PHASE_OF:
            self.render_loop(PHASE_OF[t])
        with self.left:
            if t == "goal":
                st.markdown(f"**🎯 Goal:** reproduce a claim from `{d['source']}`" + (f"  \n**Hint:** {d['focus']}" if d.get("focus") else ""))
                st.caption(f"model: {d.get('model')}")
            elif t == "step":
                self.steps = d["n"]
                self.stats()
            elif t == "thought":
                st.markdown(f"💭 _{d['text']}_")
            elif t == "tool_call" and d["name"] != "run_python":
                args = ", ".join(f"{k}={v!r}" for k, v in d["args"].items())
                st.markdown(f"{TOOL_ICONS.get(d['name'], '🔧')} **{d['name']}**  `{args[:180]}`")
            elif t == "tool_result" and d["name"] != "run_python":
                if not d["ok"]:
                    st.error(d["content"][:600])
                elif d["name"] in ("fetch_paper", "search_paper", "search_datasets", "check_environment"):
                    with st.expander("observation", expanded=False):
                        st.text(d["content"][:3000])
            elif t == "run_start":
                self.status = st.status(f"🐍 {d['purpose'] or 'running code'}", expanded=False)
                with self.status:
                    st.code(d["code"], language="python")
            elif t == "run_end":
                self.runs += 1
                if not d["ok"]:
                    self.fails += 1
                res = ", ".join(f"{k}={v:g}" for k, v in d["results"].items())
                label = f"🐍 {d['purpose']}: " + (f"✅ {res or 'ok'}" if d["ok"] else ("⏱️ timeout" if d["timed_out"] else "❌ failed"))
                if self.status is None:
                    self.status = st.status(label)
                with self.status:
                    st.text((d["stdout"] or "")[-2500:])
                    if d["stderr"].strip() and not d["ok"]:
                        st.text(d["stderr"][-2000:])
                self.status.update(label=label + f"  ({d['seconds']}s)", state="complete" if d["ok"] else "error")
                self.status = None
                self.stats()
            elif t == "reflect":
                st.warning(f"🤔 {d['fail_streak']} failures in a row. The agent is forced to stop and diagnose the root cause.")
            elif t == "install":
                st.info(f"📦 installed {d['package']}" if d["ok"] else f"📦 install of {d['package']} failed")
            elif t == "lesson":
                self.lessons.append(d["lesson"])
                st.success(f"🧠 saved lesson: {d['lesson']}")
                self.stats()
            elif t in ("nudge", "error"):
                st.warning(d.get("text", ""))
        if t == "paper":
            self.paper_ph.markdown(f'<div class="card"><h4>Paper</h4><b>{d["title"]}</b><br>{d["source"]} · {d["pages"]} pages</div>', unsafe_allow_html=True)
        elif t == "claim":
            self.claim_ph.markdown(
                f'<div class="card"><h4>Claim under test</h4>“{d["claim_text"]}”<br><br>'
                f'<b>{d["metric"]}</b> = <b>{d["reported_value"]}{d.get("unit", "")}</b> · {d["location"]}<br>'
                f'<small>{d["method"]} on {d["dataset"]}</small></div>', unsafe_allow_html=True)
        elif t == "plan":
            mark = {"done": "✅", "doing": "⏳", "todo": "⬜", "skipped": "➖"}
            items = "".join(f'<div>{mark[s["status"]]} {s["step"]}</div>' for s in d["steps"])
            self.plan_ph.markdown(f'<div class="card plan"><h4>Plan</h4>{items}</div>', unsafe_allow_html=True)
        elif t == "verdict":
            color = VERDICT_COLORS.get(d["label"], "#333")
            if d.get("measured") is not None:
                body = (f'paper <b>{d["reported"]:g}{d["unit"]}</b> → ours <b>{d["measured"]:.4g}{d["unit"]}</b>'
                        f' (gap {d["abs_gap"]:+.3g}, tolerance ±{d["tolerance"]:.3g})')
            else:
                body = d.get("note", "")
            self.verdict_ph.markdown(f'<div class="verdict" style="background:{color}"><div class="big">{d["label"]}</div>{body}</div>',
                                     unsafe_allow_html=True)


def show_outputs(run_dir: Path):
    run_dir = Path(run_dir)
    if (run_dir / "comparison.png").exists():
        st.image(str(run_dir / "comparison.png"), width=520)
    if (run_dir / "report.md").exists():
        with st.expander("📝 Full reproduction report", expanded=False):
            st.markdown((run_dir / "report.md").read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download report (HTML)", (run_dir / "report.html").read_bytes(), "repro_report.html", "text/html")
        c2.download_button("Download report (Markdown)", (run_dir / "report.md").read_bytes(), "repro_report.md")
        c3.download_button("Download trace (JSON)", (run_dir / "trace.json").read_bytes(), "trace.json", "application/json")


def replay(run_dir: Path, speed: float):
    events = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))["events"]
    view = TraceView()
    prev = 0.0
    for ev in events:
        if speed > 0:
            time.sleep(min(max(ev["t"] - prev, 0) / speed, 2.5))
        prev = ev["t"]
        view.handle(ev)
    show_outputs(run_dir)


def list_runs() -> list[Path]:
    return sorted((p for p in RUNS.glob("*") if (p / "trace.json").exists()), reverse=True)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    providers = ["anthropic"] + list(OPENAI_COMPAT_PRESETS)
    default_provider = os.getenv("REPRO_PROVIDER", "anthropic")
    provider = st.selectbox("LLM provider", providers, index=providers.index(default_provider) if default_provider in providers else 0)
    model = st.text_input("Model (blank = default)", os.getenv("REPRO_MODEL", ""))
    api_key = st.text_input("API key (blank = from .env)", type="password")
    base_url = st.text_input("Base URL override", "") if provider != "anthropic" else ""
    st.divider()
    max_steps = st.slider("Max agent steps", 10, 80, 40)
    max_runs = st.slider("Max code runs", 3, 30, 14)
    run_timeout = st.slider("Default seconds per run", 60, 1200, 300, step=60)
    max_minutes = st.slider("Total time budget (min)", 5, 90, 30)
    online = st.toggle("Online dataset search (OpenML, Hugging Face)", True)
    allow_install = st.toggle("Allow pip installs (allowlisted)", True)

st.title("⚖️ Repro")
st.caption("Give it a paper. It picks a claim, re-runs the experiment itself, and tells you if the number holds up.")

tab_run, tab_replay, tab_memory = st.tabs(["▶️ Run", "🎬 Replay a run", "🧠 Memory"])

with tab_run:
    c1, c2 = st.columns([3, 2])
    source = c1.text_input("arXiv id, arXiv/PDF URL, or path", placeholder="e.g. 2106.11959  or  examples/toy_paper.txt")
    upload = c2.file_uploader("…or upload a PDF", type=["pdf"])
    focus = st.text_input("Optional focus hint", placeholder="e.g. 'Table 2, XGBoost on California Housing'")
    go = st.button("Reproduce it", type="primary")

    if go:
        if upload is not None:
            up = Path("uploads"); up.mkdir(exist_ok=True)
            path = up / upload.name
            path.write_bytes(upload.getvalue())
            source = str(path)
        if not source.strip():
            st.warning("Enter a paper first.")
            st.stop()
        try:
            llm = make_llm(provider, model or None, api_key or None, base_url or None)
        except Exception as e:
            st.error(f"LLM setup failed: {e}")
            st.stop()
        view = TraceView()
        agent = ReproAgent(llm, runs_dir=RUNS, memory_path=MEMORY, max_steps=max_steps, max_runs=max_runs,
                           run_timeout=run_timeout, max_minutes=max_minutes, online=online,
                           allow_install=allow_install, on_event=view.handle)
        try:
            out = agent.run(source.strip(), focus)
        except Exception as e:
            st.exception(e)
            st.stop()
        st.session_state["last_run"] = str(out.workdir)
        show_outputs(out.workdir)
    elif st.session_state.get("last_run"):
        st.info("Showing your last run.")
        replay(Path(st.session_state["last_run"]), speed=0)

with tab_replay:
    runs = list_runs()
    if not runs:
        st.write("No saved runs yet. Finished runs appear here and can be replayed for demo recordings.")
    else:
        pick = st.selectbox("Saved run", runs, format_func=lambda p: p.name)
        speed = st.select_slider("Replay speed", options=[0.5, 1.0, 2.0, 4.0, 8.0, 0.0], value=4.0,
                                 format_func=lambda s: "instant" if s == 0 else f"{s:g}x")
        if st.button("Replay"):
            replay(pick, speed)

with tab_memory:
    mem = LessonMemory(MEMORY)
    items = mem.all()
    st.write("Lessons the agent saved from its own failures. They are injected into every future run.")
    for it in reversed(items):
        st.markdown(f"- {it['lesson']}  \n  <small>from {it.get('source', '?')}</small>", unsafe_allow_html=True)
    if not items:
        st.caption("Empty so far.")
    if items and st.button("Clear memory"):
        MEMORY.unlink(missing_ok=True)
        st.rerun()
