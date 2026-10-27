from embedify.chunkers import MarkdownChunker
from embedify.config import ChunkerConfig
from embedify.types import Modality, Unit


def _md_unit(text):
    return Unit(content=text, modality=Modality.TEXT, source_uri="doc.md",
                metadata={"format": "md"})


def test_markdown_heading_path_prepended(fixtures_dir):
    chunks = list(MarkdownChunker().chunk([_md_unit((fixtures_dir / "sample.md").read_text())]))
    by_path = {c.metadata.get("heading_path"): c for c in chunks}
    assert "Guide > Config > Endpoint" in by_path
    endpoint = by_path["Guide > Config > Endpoint"]
    assert endpoint.content.startswith("Guide > Config > Endpoint")
    assert "Ollama" in endpoint.content


def test_markdown_fence_never_split_and_not_a_heading(fixtures_dir):
    chunks = list(MarkdownChunker(ChunkerConfig(max_tokens=10)).chunk(
        [_md_unit((fixtures_dir / "sample.md").read_text())]))
    paths = {c.metadata.get("heading_path") for c in chunks}
    assert not any(p and "also not a heading" in p for p in paths)
    fence_chunks = [c for c in chunks if "not_a_heading" in c.content]
    assert fence_chunks
    assert all("```" in c.content for c in fence_chunks)
    # fence kept whole in a single chunk
    assert sum("def not_a_heading" in c.content for c in chunks) == 1


def test_markdown_via_pipeline_is_default(fixtures_dir, hash_config, hash_embedder):
    import embedify
    result = embedify.embed(fixtures_dir / "sample.md", config=hash_config,
                            embedder=hash_embedder)
    assert any(r.metadata.get("heading_path") == "Guide > Usage" for r in result.records)
