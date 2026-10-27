"""Modality detection via extension, MIME sniffing and content inspection.

Detection is overridable: pass ``detector=`` to :func:`embedify.embed`, or
register a rule with :func:`register_extension`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embedify.types import Modality

EXTENSION_MAP: dict[str, Modality] = {
    ".txt": Modality.TEXT,
    ".md": Modality.TEXT,
    ".markdown": Modality.TEXT,
    ".rst": Modality.TEXT,
    ".log": Modality.TEXT,
    ".html": Modality.TEXT,
    ".htm": Modality.TEXT,
    ".eml": Modality.TEXT,
    ".pdf": Modality.DOCUMENT,
    ".py": Modality.CODE,
    ".pyi": Modality.CODE,
    ".js": Modality.CODE,
    ".jsx": Modality.CODE,
    ".ts": Modality.CODE,
    ".tsx": Modality.CODE,
    ".java": Modality.CODE,
    ".go": Modality.CODE,
    ".rs": Modality.CODE,
    ".c": Modality.CODE,
    ".h": Modality.CODE,
    ".cpp": Modality.CODE,
    ".cc": Modality.CODE,
    ".hpp": Modality.CODE,
    ".rb": Modality.CODE,
    ".php": Modality.CODE,
    ".sh": Modality.CODE,
    ".sql": Modality.CODE,
    ".r": Modality.CODE,
    ".swift": Modality.CODE,
    ".kt": Modality.CODE,
    ".scala": Modality.CODE,
    ".csv": Modality.TABULAR,
    ".tsv": Modality.TABULAR,
    ".parquet": Modality.TABULAR,
    ".json": Modality.TABULAR,
    ".jsonl": Modality.TABULAR,
    ".xlsx": Modality.TABULAR,
    ".xlsm": Modality.TABULAR,
    ".db": Modality.TABULAR,
    ".sqlite": Modality.TABULAR,
    ".sqlite3": Modality.TABULAR,
    ".prom": Modality.TIMESERIES,
    ".png": Modality.IMAGE,
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".webp": Modality.IMAGE,
    ".gif": Modality.IMAGE,
    ".mp3": Modality.AUDIO,
    ".wav": Modality.AUDIO,
    ".flac": Modality.AUDIO,
    ".ogg": Modality.AUDIO,
    ".mp4": Modality.VIDEO,
    ".mov": Modality.VIDEO,
    ".mkv": Modality.VIDEO,
    ".webm": Modality.VIDEO,
}

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go",
    ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".sh": "shell", ".sql": "sql", ".r": "r",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
}


def register_extension(ext: str, modality: Modality) -> None:
    EXTENSION_MAP[ext.lower() if ext.startswith(".") else f".{ext.lower()}"] = modality


@dataclass(frozen=True)
class Detection:
    modality: Modality
    source_uri: str
    mime_type: str | None = None
    language: str | None = None
    detail: str = ""


def _sniff(header: bytes) -> Modality | None:
    if header.startswith(b"%PDF"):
        return Modality.DOCUMENT
    if header.startswith(b"\x89PNG") or header.startswith(b"\xff\xd8\xff"):
        return Modality.IMAGE
    if header.startswith(b"PK\x03\x04"):
        return None  # zip-family: let extension decide (xlsx/docx/epub/...)
    return None


def _looks_like_text(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return sum(b < 9 or b > 126 for b in sample) / len(sample) < 0.05


class Detector:
    """Default detector. Subclass/replace to customise routing."""

    def detect(self, path: str | Path) -> Detection:
        p = Path(path)
        uri = str(p)
        suffix = p.suffix.lower()
        try:
            with p.open("rb") as fh:
                header = fh.read(8192)
        except OSError:
            header = b""
        sniffed = _sniff(header)
        if sniffed is not None:
            return Detection(sniffed, uri, language=LANGUAGE_MAP.get(suffix),
                             detail="magic-bytes")
        if suffix in EXTENSION_MAP:
            return Detection(EXTENSION_MAP[suffix], uri,
                             language=LANGUAGE_MAP.get(suffix), detail="extension")
        if header and _looks_like_text(header):
            return Detection(Modality.TEXT, uri, detail="content-text")
        return Detection(Modality.UNKNOWN, uri, detail="fallback-unknown")

    def detect_bytes(self, data: bytes, *, name: str = "<bytes>") -> Detection:
        sniffed = _sniff(data[:8192])
        if sniffed is not None:
            return Detection(sniffed, name, detail="magic-bytes")
        suffix = Path(name).suffix.lower()
        if suffix in EXTENSION_MAP:
            return Detection(EXTENSION_MAP[suffix], name,
                             language=LANGUAGE_MAP.get(suffix), detail="extension")
        if _looks_like_text(data[:8192]):
            return Detection(Modality.TEXT, name, detail="content-text")
        return Detection(Modality.UNKNOWN, name, detail="fallback-unknown")


def detect(path: str | Path) -> Detection:
    return Detector().detect(path)
