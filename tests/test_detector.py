from embedify.detector import Detector
from embedify.types import Modality


def test_detect_text(fixtures_dir):
    assert Detector().detect(fixtures_dir / "sample.txt").modality == Modality.TEXT


def test_detect_code_language(fixtures_dir):
    det = Detector().detect(fixtures_dir / "sample.py")
    assert det.modality == Modality.CODE
    assert det.language == "python"


def test_detect_unknown_binary(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\xff\xfe" * 20)
    assert Detector().detect(p).modality == Modality.UNKNOWN


def test_detect_bytes_pdf(tmp_path):
    det = Detector().detect_bytes(b"%PDF-1.4 fake", name="x.bin")
    assert det.modality == Modality.DOCUMENT
