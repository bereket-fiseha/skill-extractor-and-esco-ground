"""Turn resume files / pasted text into candidate skill phrases."""
from __future__ import annotations

import io
import re

STOPWORDS = frozenset(
    """a an the and or of to in on for with by at from as is are was were be been being this that these those
    it its i we you he she they my our your their me us them using use used also etc via per into over under
    including include includes such than then so if not no can will would should could may might have has had
    do does did within across through while during about more most other various several many""".split()
)
SKILL_HEADER = re.compile(
    r"^\s*(technical\s+|key\s+|core\s+|professional\s+)?(skills?|competenc(?:y|ies)|technologies|expertise|tools)\b.*$",
    re.I,
)
NEXT_HEADER = re.compile(
    r"^\s*(work\s+)?(experience|education|projects?|certifications?|awards?|publications?|summary|profile|"
    r"employment|references|interests|languages|volunteer\w*)\b.{0,30}$",
    re.I,
)


def read_resume(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        import docx

        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def _dedupe(items) -> list[str]:
    seen, out = set(), []
    for s in items:
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def split_skill_list(text: str) -> list[str]:
    """Split a pasted list (commas, semicolons, bullets, newlines) into clean phrases."""
    parts = re.split(r"[\n,;•·●▪■|/]+|\s[-–—]\s|\s{3,}", text)
    return _dedupe(p.strip(" \t-–—*:.()") for p in parts if 1 < len(p.strip()) <= 80)


def _skills_section(text: str) -> str | None:
    grabbed, active = [], False
    for line in text.splitlines():
        if SKILL_HEADER.match(line) and len(line) < 60:
            active = True
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip():
                grabbed.append(rest)
            continue
        if active and NEXT_HEADER.match(line):
            active = False
        if active:
            grabbed.append(line)
    return "\n".join(grabbed) if grabbed else None


def phrases_from_resume(text: str, max_ngram: int = 4) -> tuple[list[str], str]:
    """Return (candidate phrases, strategy used).

    1. An explicit "Skills" section -> its items are used as-is (precise).
    2. Otherwise every 1..max_ngram word span of each clause that neither starts nor
       ends with a stop-word (broad; the similarity threshold does the filtering).
    """
    section = _skills_section(text)
    if section:
        items = split_skill_list(section)
        if len(items) >= 3:
            return items, "skills section"

    spans: list[str] = []
    for clause in re.split(r"[\n.;:,•·●▪■|()]+", text):
        words = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]*", clause)
        for n in range(1, max_ngram + 1):
            for i in range(len(words) - n + 1):
                gram = words[i : i + n]
                if gram[0].lower() in STOPWORDS or gram[-1].lower() in STOPWORDS:
                    continue
                if n == 1 and len(gram[0]) < 3:
                    continue
                spans.append(" ".join(gram))
    return _dedupe(spans), "n-gram scan"
