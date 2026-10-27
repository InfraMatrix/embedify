"""Secret scanning *before* embedding.

Files that look like secret stores (.env, private keys, credentials) are
excluded wholesale; other files are scanned with high-precision patterns
and excluded entirely if anything matches. Findings never contain the
secret value — only kind + line — so they are safe to log/report.
"""

from __future__ import annotations

import re
from pathlib import Path

from embedify.types import SecretFinding

SECRET_FILENAMES = {"credentials", "credentials.json", "secrets.json", "secrets.yaml",
                    "secrets.yml", "id_rsa", "id_ed25519", "id_dsa", ".npmrc", ".pypirc"}
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("generic-assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd)"
        r"\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']")),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._=-]{20,}")),
]


def is_secret_filename(path: Path) -> bool:
    name = path.name.lower()
    return (
        name == ".env" or name.startswith(".env.")
        or name in SECRET_FILENAMES
        or name.endswith(SECRET_SUFFIXES)
    )


def scan_text(text: str, *, source_uri: str) -> list[SecretFinding]:
    """Scan text; returns findings (kind + 1-based line, never the value)."""
    findings: list[SecretFinding] = []
    seen: set[tuple[str, int]] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in _PATTERNS:
            if pattern.search(line) and (kind, lineno) not in seen:
                seen.add((kind, lineno))
                findings.append(SecretFinding(source_uri=source_uri, kind=kind, line=lineno))
    return findings
