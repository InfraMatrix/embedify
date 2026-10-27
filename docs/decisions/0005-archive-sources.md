# 0005 — Archive sources

Status: accepted (2026-10-27).

`.tar`, `.tar.gz`, `.tgz`, and `.zip` are accepted anywhere a file source
is. They are extracted to a temp directory (cleaned up at end of run) and
then processed **exactly like a directory** — same `.gitignore` handling,
vendored/generated/lock/binary skips, secret scanning, chunking, and
per-file error isolation.

Safety (any violation rejects the whole archive, reported as a per-file
error): no absolute or `..` member paths, no member escaping the
extraction root, no symlinks/hardlinks/devices/fifos, and size limits
(per-member 100 MiB, total 1 GiB, 10k members) as zip-bomb guards.

Identity: records keep the archive in `source_uri` as
`archive.tar.gz!inner/path.py`; locators are relative to the inner file,
and code `path` metadata is the archive-inner path (git provenance is
null inside archives — there is no repo).

Archives found inside directory walks — and archives inside archives —
are expanded recursively (max nesting depth 5), chaining URIs:
``outer.zip!inner.zip!deep/file.py``.
