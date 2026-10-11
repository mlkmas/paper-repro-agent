"""Write the run's outputs: report.md, report.html (self-contained), trace.json, transcript.json."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from .verdict import BLOCKED, NOT_REPRODUCED, PARTIAL, REPRODUCED

COLORS = {REPRODUCED: "#1f8a4c", PARTIAL: "#b7791f", NOT_REPRODUCED: "#c0392b", BLOCKED: "#5f6b7a"}


def comparison_chart(outcome, path: Path) -> Path | None:
    v = outcome.verdict
    if v.label == BLOCKED or v.reported is None or v.measured is None:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    c = outcome.state.claim
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
    bars = ax.bar(["Paper (reported)", "Repro (measured)"], [v.reported, v.measured],
                  color=["#9aa5b1", COLORS[v.label]], width=0.55)
    ax.axhspan(v.reported - v.tolerance, v.reported + v.tolerance, color="#9aa5b1", alpha=0.18, label="tolerance band")
    for b, val in zip(bars, [v.reported, v.measured]):
        ax.annotate(f"{val:.4g}{v.unit}", (b.get_x() + b.get_width() / 2, val), ha="center", va="bottom",
                    fontsize=9, xytext=(0, 3), textcoords="offset points")
    lo = min(v.reported, v.measured) - 4 * v.tolerance
    hi = max(v.reported, v.measured) + 4 * v.tolerance
    if v.unit == "%":
        hi = min(hi, 100.5)
    ax.set_ylim(max(lo, 0) if min(v.reported, v.measured) >= 0 else lo, hi)
    ax.set_ylabel(f"{c['metric']} ({v.unit})" if v.unit else c["metric"])
    ax.set_title(v.label, color=COLORS[v.label], fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def markdown(outcome) -> str:
    s, v = outcome.state, outcome.verdict
    f = s.finished or {}
    c = s.claim or {}
    paper = s.paper
    lines = [f"# Reproduction report: {paper.title if paper else 'unknown paper'}", "",
             f"**Source:** {paper.source if paper else '?'}  ", f"**Run:** `{outcome.run_id}`", "",
             f"## Verdict: {v.label}", "", v.one_line(), ""]
    if v.note and v.label != BLOCKED:
        lines += [f"_Note: {v.note}_", ""]
    if c:
        lines += ["## Claim tested", "", f"> {c.get('claim_text', '')}", "",
                  f"- **Location:** {c.get('location', '')}", f"- **Method:** {c.get('method', '')}",
                  f"- **Dataset:** {c.get('dataset', '')}",
                  f"- **Metric:** {c.get('metric', '')} (reported {c.get('reported_value')}{c.get('unit', '')}, "
                  f"{'higher' if c.get('higher_is_better') else 'lower'} is better)",
                  f"- **Paper's setup:** {c.get('setup', '')}"]
        if c.get("why_this_claim"):
            lines.append(f"- **Why this claim:** {c['why_this_claim']}")
        lines.append("")
    lines += ["## What the agent did", "", f.get("explanation", ""), ""]
    if f.get("blocked_reason") and f["blocked_reason"] not in v.one_line():
        lines += [f"**Blocked because:** {f['blocked_reason']}", ""]
    if f.get("deviations"):
        lines += ["## Deviations from the paper", ""] + [f"- {d}" for d in f["deviations"]] + [""]
    if s.plan:
        mark = {"done": "x", "doing": "~", "todo": " ", "skipped": "-"}
        lines += ["## Plan", ""] + [f"- [{mark[p['status']]}] {p['step']}" for p in s.plan] + [""]
    if s.runs:
        lines += ["## Experiment runs", "", "| # | Purpose | Status | Time | Results |", "|---|---|---|---|---|"]
        for r in s.runs:
            st = "timeout" if r["timed_out"] else ("ok" if r["ok"] else "failed")
            res = ", ".join(f"{k}={val:g}" for k, val in r["results"].items()) or "-"
            lines.append(f"| {r['script']} | {r['purpose'].replace('|', '/')} | {st} | {r['seconds']:.0f}s | {res} |")
        lines.append("")
        fails = sum(1 for r in s.runs if not r["ok"])
        if fails:
            lines += [f"The agent hit {fails} failed run(s) and recovered by diagnosing the errors.", ""]
    if s.claim_history:
        lines += ["## Claims considered and dropped", ""] + [f"- {h.get('claim_text')} ({h.get('location')})" for h in s.claim_history] + [""]
    if s.installs:
        lines += ["## Packages installed during the run", "", ", ".join(s.installs), ""]
    if s.lessons_saved:
        lines += ["## Lessons saved to long-term memory", ""] + [f"- {l}" for l in s.lessons_saved] + [""]
    lines += ["## Reproduce this reproduction", "",
              f"All scripts are in `{outcome.workdir / 'experiment'}` (`step_XX.py`, run in order).", ""]
    if outcome.stopped_reason:
        lines += [f"_Run stopped early: {outcome.stopped_reason}._", ""]
    return "\n".join(lines)


def to_html(md_text: str, outcome, chart: Path | None) -> str:
    """Tiny markdown-to-HTML so the report is one shareable file with no extra dependency."""
    import re

    out, in_list, in_table = [], False, False
    for raw in md_text.splitlines():
        line = html.escape(raw)
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        line = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<em>\1</em>", line)
        if in_list and not raw.startswith("- "):
            out.append("</ul>"); in_list = False
        if in_table and not raw.startswith("|"):
            out.append("</table>"); in_table = False
        if raw.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif raw.startswith("## Verdict"):
            color = COLORS.get(outcome.verdict.label, "#333")
            out.append(f'<h2 class="verdict" style="background:{color}">{line[3:]}</h2>')
            if chart:
                b64 = base64.b64encode(chart.read_bytes()).decode()
                out.append(f'<img alt="comparison chart" src="data:image/png;base64,{b64}">')
        elif raw.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif raw.startswith("> "):
            out.append(f"<blockquote>{line[5:]}</blockquote>")
        elif raw.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{line[2:]}</li>")
        elif raw.startswith("|"):
            if set(raw.replace("|", "").strip()) <= {"-"}:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>"); in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        elif raw.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    if in_table:
        out.append("</table>")
    style = """body{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#1d2330}
h1{font-size:24px}h2{font-size:18px;margin-top:28px}.verdict{color:#fff;padding:10px 14px;border-radius:8px}
blockquote{border-left:4px solid #9aa5b1;margin:0;padding:6px 14px;background:#f4f6f8}img{max-width:100%;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d9dee4;padding:6px 8px;text-align:left}
code{background:#f0f2f5;padding:1px 4px;border-radius:4px}"""
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Repro report</title><style>{style}</style></head><body>{''.join(out)}</body></html>"


def write_outputs(outcome, messages: list[dict]) -> Path:
    wd = Path(outcome.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    chart = comparison_chart(outcome, wd / "comparison.png")
    md = markdown(outcome)
    (wd / "report.md").write_text(md, encoding="utf-8")
    (wd / "report.html").write_text(to_html(md, outcome, chart), encoding="utf-8")
    summary = {"run_id": outcome.run_id, "verdict": outcome.verdict.to_dict(), "claim": outcome.state.claim,
               "finished": outcome.state.finished, "plan": outcome.state.plan,
               "paper": {"title": outcome.state.paper.title, "source": outcome.state.paper.source} if outcome.state.paper else None}
    (wd / "trace.json").write_text(json.dumps({"summary": summary, "events": outcome.trace}, indent=1, default=str), encoding="utf-8")
    (wd / "transcript.json").write_text(json.dumps(messages, indent=1, default=str), encoding="utf-8")
    return wd / "report.md"
