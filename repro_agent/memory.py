"""Long-term memory: lessons the agent saves from its own failures, reused in future runs."""
from __future__ import annotations

import json
import time
from pathlib import Path


class LessonMemory:
    def __init__(self, path: Path, max_items: int = 30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out[-self.max_items:]

    def add(self, lesson: str, source: str = "") -> str:
        lesson = " ".join(lesson.split())[:400]
        if not lesson:
            return "Empty lesson ignored."
        if any(l["lesson"].lower() == lesson.lower() for l in self.all()):
            return "Already known."
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"lesson": lesson, "source": source, "ts": int(time.time())}) + "\n")
        return "Saved. Future runs will see this lesson."

    def as_prompt(self) -> str:
        items = self.all()
        if not items:
            return "(no lessons yet)"
        return "\n".join(f"- {l['lesson']}" for l in items)
