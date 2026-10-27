"""Archive sources: safe extraction, directory semantics, archive!inner URIs."""

import io
import tarfile
import zipfile

import pytest

import embedify
from embedify.archives import archive_uri, extract_archive, is_archive
from embedify.config import EmbedderConfig, EmbedifyConfig, SinkConfig
from embedify.errors import LoaderError


def _cfg() -> EmbedifyConfig:
    return EmbedifyConfig(
        embedders={"text": EmbedderConfig(backend="hash", model="hash", dim=32)},
        sink=SinkConfig(type="memory"))


def _make_zip(path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def _make_targz(path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_is_archive_and_uri():
    from pathlib import Path

    assert is_archive(Path("a.tar.gz")) and is_archive(Path("a.tgz"))
    assert is_archive(Path("a.tar")) and is_archive(Path("a.zip"))
    assert not is_archive(Path("a.txt"))
    assert archive_uri("x.tar.gz", "src/app.py") == "x.tar.gz!src/app.py"


def test_zip_processed_like_directory_with_archive_uris(tmp_path):
    archive = tmp_path / "code.zip"
    _make_zip(archive, {
        "src/app.py": "def add(a, b):\n    return a + b\n",
        "docs/notes.txt": "hello archive world",
        ".gitignore": "ignored.py\n",
        "ignored.py": "def ignored():\n    return 0\n",
        "package-lock.json": "{}",
    })
    result = embedify.embed(archive, config=_cfg())
    assert not result.errors
    uris = {r.source_uri for r in result.records}
    assert f"{archive}!src/app.py" in uris
    assert f"{archive}!docs/notes.txt" in uris
    assert not any("ignored.py" in u or "package-lock" in u for u in uris)
    code = next(r for r in result.records if r.source_uri.endswith("!src/app.py"))
    assert code.metadata["path"] == "src/app.py"
    assert code.metadata["repo"] is None
    assert code.locator.line_start is not None


def test_targz_processed_with_archive_uris(tmp_path):
    archive = tmp_path / "data.tar.gz"
    _make_targz(archive, {"a.txt": "tar archive content here"})
    result = embedify.embed(str(archive), config=_cfg())
    assert not result.errors
    assert {r.source_uri for r in result.records} == {f"{archive}!a.txt"}


def test_plain_tar_and_tgz(tmp_path):
    for name, mode in (("x.tar", "w"), ("x.tgz", "w:gz")):
        archive = tmp_path / name
        with tarfile.open(archive, mode) as tf:
            data = b"some tar text"
            info = tarfile.TarInfo("f.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        result = embedify.embed(archive, config=_cfg())
        assert not result.errors and result.records


def test_zip_path_traversal_rejected_and_nothing_written(tmp_path):
    archive = tmp_path / "evil.zip"
    _make_zip(archive, {"../evil.txt": "pwned", "ok.txt": "fine"})
    result = embedify.embed(archive, config=_cfg())
    assert result.records == []
    assert result.errors and "Unsafe archive member" in result.errors[0].error
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()


def test_tar_symlink_rejected(tmp_path):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link.py")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    result = embedify.embed(archive, config=_cfg())
    assert result.records == []
    assert result.errors and "link" in result.errors[0].error.lower()


def test_extract_archive_size_and_count_limits(tmp_path):
    archive = tmp_path / "big.zip"
    _make_zip(archive, {"a.txt": "x" * 500, "b.txt": "y" * 500})
    with pytest.raises(LoaderError, match="size limit"):
        extract_archive(archive, tmp_path / "out1", max_total_bytes=600)
    with pytest.raises(LoaderError, match="size limit"):
        extract_archive(archive, tmp_path / "out2", max_member_bytes=100)
    with pytest.raises(LoaderError, match="more than"):
        extract_archive(archive, tmp_path / "out3", max_files=1)


def test_secret_inside_archive_excluded_with_archive_uri(tmp_path):
    archive = tmp_path / "s.zip"
    _make_zip(archive, {
        ".env": "API_KEY=whatever123456\n",
        "clean.txt": "perfectly clean text",
    })
    result = embedify.embed(archive, config=_cfg())
    uris = {r.source_uri for r in result.records}
    assert uris == {f"{archive}!clean.txt"}
    assert result.findings and result.findings[0].source_uri == f"{archive}!.env"


def test_corrupt_archive_is_per_file_error_not_crash(tmp_path):
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"this is not a zip")
    result = embedify.embed(archive, config=_cfg())
    assert result.records == []
    assert result.errors and "zip" in result.errors[0].error.lower()


def test_zip_bomb_rejected_by_configurable_total_limit(tmp_path):
    # Classic zip bomb shape: tiny on disk, huge when extracted. The total
    # extracted-bytes limit is configurable via ArchiveConfig.
    import zipfile as _zf

    from embedify.config import ArchiveConfig

    archive = tmp_path / "bomb.zip"
    with _zf.ZipFile(archive, "w", compression=_zf.ZIP_DEFLATED) as zf:
        zf.writestr("zeros.bin", b"\x00" * 2_000_000)
    assert archive.stat().st_size < 20_000  # highly compressible payload
    cfg = EmbedifyConfig(
        embedders={"text": EmbedderConfig(backend="hash", model="hash", dim=32)},
        sink=SinkConfig(type="memory"),
        archive=ArchiveConfig(max_total_bytes=100_000))
    result = embedify.embed(archive, config=cfg)
    assert result.records == []
    assert result.errors and "size limit" in result.errors[0].error


def test_zip_symlink_entry_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("evil.py")
        info.create_system = 3  # unix
        info.external_attr = 0o120777 << 16  # symlink file type
        zf.writestr(info, "/etc/passwd")
        zf.writestr("ok.txt", "fine")
    result = embedify.embed(archive, config=_cfg())
    assert result.records == []
    assert result.errors and "symlink" in result.errors[0].error.lower()


def test_archive_inside_directory_walk_is_expanded(tmp_path):
    _make_zip(tmp_path / "nested.zip", {"in.txt": "from inside a directory walk"})
    (tmp_path / "plain.txt").write_text("plain directory file")
    result = embedify.embed(tmp_path, config=_cfg())
    assert not result.errors
    uris = {r.source_uri for r in result.records}
    assert any(u.endswith("nested.zip!in.txt") for u in uris)
    assert any(u.endswith("plain.txt") for u in uris)


def test_archive_inside_archive_chains_uris(tmp_path):
    import io as _io

    inner_buf = _io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as zf:
        zf.writestr("deep/code.py", "def deep():\n    return 1\n")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("inner.zip", inner_buf.getvalue())
        zf.writestr("top.txt", "outer level text")
    result = embedify.embed(outer, config=_cfg())
    assert not result.errors
    uris = {r.source_uri for r in result.records}
    assert f"{outer}!inner.zip!deep/code.py" in uris
    assert f"{outer}!top.txt" in uris


def test_archive_nesting_depth_is_capped(tmp_path):
    import io as _io

    payload: bytes = _io.BytesIO(b"bottom text").getvalue()
    # Wrap "level.zip" around the payload MAX_NESTING_DEPTH+2 times.
    for _ in range(7):
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("level.zip", payload)
        payload = buf.getvalue()
    archive = tmp_path / "deep.zip"
    archive.write_bytes(payload)
    result = embedify.embed(archive, config=_cfg())
    assert result.records == []
    assert result.errors and "nesting" in result.errors[0].error
