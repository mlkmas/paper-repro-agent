"""Load a paper (arXiv id/URL, PDF URL, or local PDF/text file) and search inside it."""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import requests

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
UA = {"User-Agent": "paper-repro-agent/1.0 (hackathon project)"}


@dataclass
class Paper:
    source: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    pages: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)

    def overview(self, max_headings: int = 40) -> str:
        heads = find_headings(self.pages)[:max_headings]
        lines = [
            f"Title: {self.title or '(unknown)'}",
            f"Authors: {', '.join(self.authors[:8]) or '(unknown)'}",
            f"Pages: {len(self.pages)}   Characters: {len(self.text):,}",
            "",
            "Abstract:",
            (self.abstract or (self.pages[0][:1500] if self.pages else "(empty)")).strip(),
            "",
            "Detected section headings (page: heading):",
        ]
        lines += [f"  p{p}: {h}" for p, h in heads] or ["  (none detected)"]
        tables = [(i + 1, m) for i, pg in enumerate(self.pages) for m in re.findall(r"\bTable\s+\d+[:.].{0,90}", pg)]
        if tables:
            lines += ["", "Table captions:"] + [f"  p{p}: {t.strip()}" for p, t in tables[:25]]
        return "\n".join(lines)

    def search(self, query: str, window: int = 700, max_hits: int = 6) -> str:
        """Rank passages by how many query terms they contain (cheap, robust, no embeddings)."""
        terms = [t for t in re.findall(r"[\w\-\.%]+", query.lower()) if len(t) > 1]
        if not terms:
            return "Empty query."
        hits = []
        step = window // 2
        for pno, page in enumerate(self.pages, start=1):
            low = page.lower()
            for start in range(0, max(1, len(page) - step), step):
                chunk = low[start:start + window]
                score = sum(chunk.count(t) for t in terms) + 3 * sum(1 for t in set(terms) if t in chunk)
                if score:
                    hits.append((score, pno, start))
        if not hits:
            return f"No passages matched {terms}. Try different keywords, or read a page with page=N."
        hits.sort(reverse=True)
        chosen, out = [], []
        for score, pno, start in hits:
            if any(p == pno and abs(s - start) < window for p, s in chosen):
                continue
            chosen.append((pno, start))
            snippet = self.pages[pno - 1][start:start + window].strip()
            out.append(f"--- page {pno} (score {score}) ---\n{snippet}")
            if len(chosen) >= max_hits:
                break
        return "\n\n".join(out)

    def page(self, n: int, max_chars: int = 6000) -> str:
        if not 1 <= n <= len(self.pages):
            return f"Page {n} out of range (1..{len(self.pages)})."
        txt = self.pages[n - 1]
        return txt[:max_chars] + (f"\n...[truncated {len(txt) - max_chars} chars]" if len(txt) > max_chars else "")


def find_headings(pages: list[str]) -> list[tuple[int, str]]:
    pat = re.compile(r"^\s*((?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-H]\.)\s+[A-Z][A-Za-z0-9 ,:\-&/()]{2,70})\s*$")
    out = []
    for pno, page in enumerate(pages, start=1):
        for line in page.splitlines():
            m = pat.match(line)
            if m and not re.search(r"\d{3,}", m.group(1)):
                out.append((pno, m.group(1).strip()))
    return out


def _pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:  # broken page, keep going
            pages.append("")
    return pages


def parse_arxiv_id(source: str) -> str | None:
    s = source.strip()
    if Path(s).exists():
        return None
    if "arxiv.org" in s or re.fullmatch(r"(arxiv:)?\s*" + ARXIV_ID.pattern, s, re.I):
        m = ARXIV_ID.search(s)
        if m:
            return (m.group(1) or m.group(3)) + (m.group(2) or m.group(4) or "")
    return None


def _arxiv_metadata(arxiv_id: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    entry = root.find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        raise ValueError(f"arXiv id {arxiv_id} not found")
    clean = lambda s: re.sub(r"\s+", " ", s or "").strip()
    return {
        "title": clean(entry.findtext("a:title", "", ns)),
        "abstract": clean(entry.findtext("a:summary", "", ns)),
        "authors": [clean(a.findtext("a:name", "", ns)) for a in entry.findall("a:author", ns)],
    }


def load_paper(source: str) -> Paper:
    source = source.strip()
    path = Path(source).expanduser()
    if path.exists():
        if path.suffix.lower() == ".pdf":
            pages = _pdf_pages(path.read_bytes())
        else:  # .txt / .md: split on form feeds or every ~3000 chars
            raw = path.read_text(encoding="utf-8", errors="replace")
            pages = raw.split("\f") if "\f" in raw else [raw[i:i + 3000] for i in range(0, len(raw), 3000)] or [""]
        paper = Paper(source=str(path), pages=pages)
        first = next((ln.strip() for ln in paper.text.splitlines() if len(ln.strip()) > 10), "")
        paper.title = first[:200]
        m = re.search(r"abstract[:.\s]*(.{200,2000}?)(?:\n\s*\n|\n1\.?\s+introduction)", paper.text, re.I | re.S)
        paper.abstract = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        return paper

    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        meta = _arxiv_metadata(arxiv_id)
        r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=UA, timeout=60)
        r.raise_for_status()
        return Paper(source=f"arXiv:{arxiv_id}", pages=_pdf_pages(r.content), **meta)

    if source.startswith(("http://", "https://")):
        r = requests.get(source, headers=UA, timeout=60)
        r.raise_for_status()
        if "pdf" not in r.headers.get("content-type", "") and not r.content[:5] == b"%PDF-":
            raise ValueError("URL did not return a PDF. Pass an arXiv id, a PDF URL, or a local file.")
        return Paper(source=source, pages=_pdf_pages(r.content))

    raise ValueError(f"Could not interpret '{source}' as an arXiv id, URL, or existing file path.")
