"""Archive sources (.tar, .tar.gz, .tgz, .zip): safe extraction.

Archives are extracted to a temp dir and then processed exactly like a
directory. Safety rules (a violation rejects the whole archive):

- no absolute paths, no ``..`` traversal, no escaping the extraction dir
- no symlinks/hardlinks/devices/fifos (zip symlink entries included)
- per-member, total-size, and member-count limits (zip-bomb guards)

Record ``source_uri`` keeps the archive identity:
``archive.tar.gz!inner/path.py``.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from embedify.errors import LoaderError

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")

MAX_FILES = 10_000
MAX_MEMBER_BYTES = 100 * 1024 * 1024      # 100 MiB per member
MAX_TOTAL_BYTES = 1024 * 1024 * 1024      # 1 GiB extracted total
MAX_NESTING_DEPTH = 5                     # archives inside archives, at most this deep

_CHUNK = 1024 * 1024


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def archive_uri(archive: Path | str, inner_rel: str) -> str:
    """Canonical source_uri for a member: ``archive.tar.gz!inner/path.py``."""
    return f"{archive}!{inner_rel.replace(chr(92), '/')}"


def _safe_target(dest: Path, member_name: str) -> Path:
    name = member_name.replace("\\", "/").strip("/")
    if not name:
        return dest
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise LoaderError(f"Unsafe archive member path (traversal): {member_name!r}")
    target = (dest / name).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise LoaderError(f"Unsafe archive member path (escapes target): {member_name!r}")
    return target


def _copy_limited(src: object, target: Path, member_name: str,
                  max_member_bytes: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("wb") as out:
        while True:
            chunk = src.read(_CHUNK)  # type: ignore[attr-defined]
            if not chunk:
                break
            written += len(chunk)
            if written > max_member_bytes:
                out.close()
                target.unlink(missing_ok=True)
                raise LoaderError(
                    f"Archive member {member_name!r} exceeds size limit "
                    f"({max_member_bytes} bytes)")
            out.write(chunk)
    return written


def extract_archive(path: Path, dest: Path, *, max_files: int = MAX_FILES,
                    max_member_bytes: int = MAX_MEMBER_BYTES,
                    max_total_bytes: int = MAX_TOTAL_BYTES) -> Path:
    """Extract ``path`` into ``dest`` safely; returns the extraction root."""
    dest.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith(".zip"):
        _extract_zip(path, dest, max_files, max_member_bytes, max_total_bytes)
    else:
        _extract_tar(path, dest, max_files, max_member_bytes, max_total_bytes)
    return dest


def _extract_zip(path: Path, dest: Path, max_files: int,
                 max_member_bytes: int, max_total_bytes: int) -> None:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise LoaderError(f"Could not read zip archive {path}: {exc}") from exc
    total = 0
    count = 0
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                _safe_target(dest, info.filename)
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise LoaderError(f"Refusing symlink in archive: {info.filename!r}")
            count += 1
            if count > max_files:
                raise LoaderError(f"Archive has more than {max_files} files")
            if info.file_size > max_member_bytes:
                raise LoaderError(
                    f"Archive member {info.filename!r} exceeds size limit "
                    f"({max_member_bytes} bytes)")
            target = _safe_target(dest, info.filename)
            with zf.open(info) as src:
                total += _copy_limited(src, target, info.filename, max_member_bytes)
            if total > max_total_bytes:
                raise LoaderError(f"Archive exceeds total extracted size limit "
                                  f"({max_total_bytes} bytes)")


def _extract_tar(path: Path, dest: Path, max_files: int,
                 max_member_bytes: int, max_total_bytes: int) -> None:
    try:
        with tarfile.open(path, "r:*") as tf:
            _extract_tar_members(tf, dest, max_files, max_member_bytes, max_total_bytes)
    except tarfile.TarError as exc:
        raise LoaderError(f"Could not read tar archive {path}: {exc}") from exc


def _extract_tar_members(tf: tarfile.TarFile, dest: Path, max_files: int,
                         max_member_bytes: int, max_total_bytes: int) -> None:
    total = 0
    count = 0
    for member in tf:
        if member.isdir():
            _safe_target(dest, member.name)
            continue
        if member.issym() or member.islnk():
            raise LoaderError(f"Refusing link in archive: {member.name!r}")
        if not member.isfile():
            raise LoaderError(f"Refusing special file in archive: {member.name!r}")
        count += 1
        if count > max_files:
            raise LoaderError(f"Archive has more than {max_files} files")
        if member.size > max_member_bytes:
            raise LoaderError(
                f"Archive member {member.name!r} exceeds size limit "
                f"({max_member_bytes} bytes)")
        target = _safe_target(dest, member.name)
        src = tf.extractfile(member)
        if src is None:
            continue
        with src:
            total += _copy_limited(src, target, member.name, max_member_bytes)
        if total > max_total_bytes:
            raise LoaderError(f"Archive exceeds total extracted size limit "
                              f"({max_total_bytes} bytes)")
