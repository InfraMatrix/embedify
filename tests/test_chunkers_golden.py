"""Golden tests: chunk boundaries and locators must not drift silently."""

from embedify.chunkers import PythonASTChunker, TokenTextChunker
from embedify.config import ChunkerConfig
from embedify.types import Modality, Unit


def _unit(text, **kw):
    return Unit(content=text, modality=Modality.TEXT, source_uri="mem://x", **kw)


def test_token_chunker_single_chunk_locator():
    text = "Hello world. This is a test."
    chunks = list(TokenTextChunker(ChunkerConfig(max_tokens=512)).chunk([_unit(text)]))
    assert len(chunks) == 1
    assert chunks[0].content == text


def test_token_chunker_splits_and_preserves_lines():
    text = "\n".join(f"Sentence number {i} is here." for i in range(50))
    chunks = list(TokenTextChunker(ChunkerConfig(max_tokens=20, overlap=0)).chunk([_unit(text)]))
    assert len(chunks) > 1
    for c in chunks:
        assert c.locator.line_start is not None
        assert c.locator.line_end >= c.locator.line_start
        assert c.locator.char_start is not None


def test_ast_chunker_symbols_and_lines(fixtures_dir):
    text = (fixtures_dir / "sample.py").read_text()
    unit = Unit(content=text, modality=Modality.CODE, source_uri=str(fixtures_dir / "sample.py"),
                metadata={"language": "python"})
    chunks = list(PythonASTChunker().chunk([unit]))
    by_symbol = {c.metadata["symbol"]: c for c in chunks}
    assert set(by_symbol) == {"add", "Greeter"}
    assert by_symbol["add"].locator.line_start == 4
    assert by_symbol["Greeter"].locator.line_start == 9
    assert "return a + b" in by_symbol["add"].content


def test_ast_chunker_fallback_unparseable():
    unit = Unit(content="def broken(:\n  pass", modality=Modality.CODE, source_uri="x.py",
                metadata={"language": "python"})
    chunks = list(PythonASTChunker().chunk([unit]))
    assert len(chunks) == 1
