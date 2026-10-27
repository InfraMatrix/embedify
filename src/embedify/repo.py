"""Repository traversal + git provenance.

Directory embedding is repo-aware: ``.gitignore`` is respected, vendored /
generated / minified / lock / binary files are skipped, and code records
carry git provenance (repo, branch, commit SHA, repo-relative path).
"""

from __future__ import annotations

import fnmatch
import subprocess
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from embedify.archives import is_archive

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "vendor", "third_party", "third-party",
    "dist", "build", "target", ".next", ".idea", ".tox", "Pods", ".gradle",
}
LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "pipfile.lock", "cargo.lock", "go.sum", "composer.lock", "gemfile.lock",
    "uv.lock",
}
MINIFIED_SUFFIXES = (".min.js", ".min.css", ".min.html")
GENERATED_MARKERS = ("code generated", "do not edit", "@generated", "auto-generated")


def is_lockfile(path: Path) -> bool:
    return path.name.lower() in LOCKFILES


def is_minified(path: Path) -> bool:
    return path.name.lower().endswith(MINIFIED_SUFFIXES) or path.suffix == ".map"


def is_binary_bytes(header: bytes) -> bool:
    return b"\x00" in header


def is_generated_text(sample: str) -> bool:
    head = sample[:2000].lower()
    return any(marker in head for marker in GENERATED_MARKERS)


class _Gitignore:
    """Minimal .gitignore matcher: globs, ``dir/``, ``/anchored``, ``!negation``."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.patterns: list[tuple[bool, bool, bool, str]] = []  # negated, dir_only, anchored, pat
        ignore_file = base / ".gitignore"
        if ignore_file.is_file():
            for raw in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                negated = line.startswith("!")
                line = line.lstrip("!").strip()
                dir_only = line.endswith("/")
                line = line.rstrip("/")
                anchored = "/" in line
                line = line.lstrip("/")
                if line:
                    self.patterns.append((negated, dir_only, anchored, line))

    def matches(self, rel: Path, is_dir: bool) -> bool:
        try:
            rel = rel.relative_to(self.base)
        except ValueError:
            return False
        rel_str = rel.as_posix()
        ignored = False
        for negated, dir_only, anchored, pattern in self.patterns:
            if dir_only and not is_dir:
                continue
            target = rel_str if anchored else rel.name
            candidates = [target]
            # A directory pattern also ignores everything beneath it.
            candidates += ["/".join(rel.parts[:i]) for i in range(1, len(rel.parts))]
            hit = any(
                fnmatch.fnmatchcase(c if anchored else Path(c).name, pattern)
                or fnmatch.fnmatchcase(c, pattern)
                for c in candidates
            )
            if hit:
                ignored = not negated
        return ignored


def iter_repo_files(root: Path) -> Iterator[Path]:
    """Walk ``root`` honouring .gitignore + vendored/generated/lock/binary skips."""
    root = root.resolve()
    matchers: list[_Gitignore] = [_Gitignore(root)]
    for dirpath, dirnames, filenames in _walk(root):
        # Recompute matcher stack depth-first: rebuild from path components.
        matchers = [_Gitignore(root)]
        current = root
        for part in dirpath.relative_to(root).parts:
            current = current / part
            matchers.append(_Gitignore(current))

        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS
            and not any(m.matches(dirpath / d, True) for m in matchers)
        )
        for name in sorted(filenames):
            path = dirpath / name
            if is_lockfile(path) or is_minified(path) or any(
                    m.matches(path, False) for m in matchers):
                continue
            if is_archive(path):
                # Archives are sources too: the pipeline expands them
                # (recursively) instead of skipping them as binaries.
                yield path
                continue
            try:
                with path.open("rb") as fh:
                    header = fh.read(8192)
            except OSError:
                continue
            if is_binary_bytes(header):
                continue
            if is_generated_text(header.decode("utf-8", errors="ignore")):
                continue
            yield path


def _walk(root: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames


@lru_cache(maxsize=64)
def _git_info_cached(start_dir: str) -> tuple[tuple[str, Any], ...]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", start_dir, *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    root = git("rev-parse", "--show-toplevel")
    if not root:
        return ()
    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    remote = git("config", "--get", "remote.origin.url")
    repo = (remote.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            if remote else Path(root).name)
    return (("repo", repo), ("repo_root", root), ("branch", branch), ("commit", commit))


def git_metadata(path: Path) -> dict[str, Any]:
    """Git provenance for a file: repo/branch/commit/path (repo-relative).

    Outside a git repo: ``repo``/``branch``/``commit`` are None and ``path``
    is the file name. Never raises — provenance is best-effort metadata.
    """
    p = path.resolve()
    info = dict(_git_info_cached(str(p.parent)))
    if not info:
        return {"repo": None, "branch": None, "commit": None, "path": p.name}
    root = Path(str(info.pop("repo_root")))
    try:
        info["path"] = p.relative_to(root).as_posix()
    except ValueError:
        info["path"] = p.name
    return info
