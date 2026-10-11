"""Deterministic verdict: the LLM measures, plain code judges.

Keeping the final call out of the model's hands means the agent cannot
talk itself into "reproduced" when the numbers say otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

REPRODUCED = "REPRODUCED"
PARTIAL = "PARTIALLY REPRODUCED"
NOT_REPRODUCED = "NOT REPRODUCED"
BLOCKED = "COULD NOT ATTEMPT"


@dataclass
class Verdict:
    label: str
    reported: float | None
    measured: float | None
    unit: str
    higher_is_better: bool | None
    abs_gap: float | None
    rel_gap: float | None
    tolerance: float | None
    note: str

    def to_dict(self) -> dict:
        return asdict(self)

    def one_line(self) -> str:
        if self.label == BLOCKED:
            return f"{self.label}: {self.note}"
        u = self.unit if self.unit not in ("", "none") else ""
        return (f"{self.label}: paper {self.reported:g}{u} vs ours {self.measured:g}{u} "
                f"(gap {self.abs_gap:+.4g}{u}, tolerance ±{self.tolerance:.4g}{u})")


def default_tolerance(reported: float, unit: str) -> float:
    u = (unit or "").lower()
    if u in ("%", "percent", "pct", "points"):
        return 1.5  # 1.5 percentage points
    if 0 < abs(reported) <= 1 and u in ("", "fraction", "ratio", "accuracy", "auc", "f1", "none"):
        return 0.015
    return 0.05 * abs(reported) if reported else 0.01  # 5% relative for errors, RMSE, etc.


def judge(reported: float | None, measured: float | None, unit: str = "", higher_is_better: bool | None = None,
          tolerance: float | None = None, blocked_reason: str | None = None) -> Verdict:
    if blocked_reason or reported is None or measured is None:
        return Verdict(BLOCKED, reported, measured, unit, higher_is_better, None, None, None,
                       blocked_reason or "missing reported or measured value")
    note = ""
    # Common slip: paper reports 93.2 (%) and code prints 0.932.
    if (unit or "").strip() in ("%", "percent") and 0 <= measured <= 1 < reported <= 100:
        measured *= 100
        note = "measured value was a fraction; converted to percent. "
    tol = tolerance if tolerance is not None else default_tolerance(reported, unit)
    gap = measured - reported
    rel = gap / abs(reported) if reported else None
    if abs(gap) <= tol:
        label = REPRODUCED
    elif higher_is_better is True and gap > 0 or higher_is_better is False and gap < 0:
        label = REPRODUCED
        note += "our result is better than the paper's claim. "
    elif abs(gap) <= 2 * tol:
        label = PARTIAL
    else:
        label = NOT_REPRODUCED
    return Verdict(label, reported, measured, unit, higher_is_better, gap, rel, tol, note.strip())
