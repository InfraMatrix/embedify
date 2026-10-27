"""Tree-sitter code chunking: symbols, context header, oversized splitting."""

import os

import pytest

from embedify.chunkers import TreeSitterChunker, _ts_parser, get_chunker
from embedify.config import ChunkerConfig
from embedify.types import Modality, Unit

PY_SRC = '''"""Module docs."""

import os


def add(a, b):
    return a + b


class Greeter:
    def greet(self, name):
        return f"hello {name}"
'''

needs_ts = pytest.mark.skipif(_ts_parser("python") is None,
                              reason="tree-sitter python grammar not installed")


def _unit(src, name="mod.py", language="python"):
    return Unit(content=src, modality=Modality.CODE, source_uri=f"/repo/{name}",
                metadata={"language": language, "path": name})


@needs_ts
def test_treesitter_python_symbols_and_module_header():
    chunks = list(TreeSitterChunker(ChunkerConfig()).chunk([_unit(PY_SRC)]))
    kinds = {(c.metadata["symbol_name"], c.metadata["symbol_kind"]) for c in chunks}
    assert ("add", "function") in kinds
    assert ("greet", "method") in kinds
    assert ("Greeter", "class") in kinds
    assert any(c.metadata["symbol_kind"] == "module_header" for c in chunks)


@needs_ts
def test_context_header_and_parent_metadata():
    chunks = list(TreeSitterChunker(ChunkerConfig()).chunk([_unit(PY_SRC)]))
    greet = next(c for c in chunks if c.metadata["symbol_name"] == "greet")
    assert greet.metadata["parent"] == "Greeter"
    assert greet.metadata["signature"].startswith("def greet")
    assert greet.content.startswith("mod.py > Greeter > def greet")
    assert greet.locator.line_start == 11 and greet.locator.line_end == 12
    add = next(c for c in chunks if c.metadata["symbol_name"] == "add")
    assert add.content.startswith("mod.py > def add")
    assert add.locator.line_start == 6


@needs_ts
def test_oversized_symbol_split_repeats_signature():
    body = "\n".join(f"    x{i} = compute_{i}(value)" for i in range(60))
    src = f"def big(value):\n{body}\n    return value\n"
    chunks = list(TreeSitterChunker(ChunkerConfig(max_tokens=64)).chunk([_unit(src)]))
    pieces = [c for c in chunks if c.metadata["symbol_name"] == "big"]
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.content.startswith("mod.py > def big")
        assert piece.locator.line_start is not None


@pytest.mark.skipif(_ts_parser("go") is None, reason="tree-sitter go grammar not installed")
def test_treesitter_go_function():
    src = "package main\n\nfunc Add(a int, b int) int {\n\treturn a + b\n}\n"
    chunks = list(TreeSitterChunker(ChunkerConfig()).chunk(
        [_unit(src, name="main.go", language="go")]))
    fn = next(c for c in chunks if c.metadata["symbol_kind"] == "function")
    assert fn.metadata["symbol_name"] == "Add"
    assert fn.content.startswith("main.go > func Add")


def test_code_chunker_default_is_treesitter_with_fallback():
    chunker = get_chunker(Modality.CODE, ChunkerConfig())
    assert isinstance(chunker, TreeSitterChunker)
    # Without grammars installed this still yields chunks (AST/token fallback).
    chunks = list(chunker.chunk([_unit(PY_SRC)]))
    assert chunks


_GRAMMAR_SAMPLES = [
    ("python", "def add(a, b):\n    return a + b\n", "mod.py", "add"),
    ("javascript", "function add(a, b) { return a + b; }\n", "mod.js", "add"),
    ("typescript", "function add(a: number, b: number): number { return a + b; }\n",
     "mod.ts", "add"),
    ("go", "package main\n\nfunc Add(a int, b int) int { return a + b }\n", "mod.go", "Add"),
    ("rust", "fn add(a: i32, b: i32) -> i32 { a + b }\n", "mod.rs", "add"),
    ("java", "class A { int add(int a, int b) { return a + b; } }\n", "A.java", "add"),
    ("c", "int add(int a, int b) { return a + b; }\n", "mod.c", "add"),
    ("cpp", "int add(int a, int b) { return a + b; }\n", "mod.cpp", "add"),
]


@pytest.mark.parametrize(("language", "src", "name", "symbol"), _GRAMMAR_SAMPLES)
def test_every_supported_grammar_chunks(language, src, name, symbol):
    if _ts_parser(language) is None:
        pytest.skip(f"tree-sitter {language} grammar not installed")
    chunks = list(TreeSitterChunker(ChunkerConfig()).chunk(
        [_unit(src, name=name, language=language)]))
    assert symbol in {c.metadata["symbol_name"] for c in chunks}


def test_all_grammars_available_when_required_in_ci():
    # CI sets EMBEDIFY_REQUIRE_GRAMMARS=1 (with embedify[code] installed):
    # there, a missing grammar must FAIL, never silently skip.
    if os.environ.get("EMBEDIFY_REQUIRE_GRAMMARS") != "1":
        pytest.skip("grammar strictness only enforced in CI")
    from embedify.chunkers import _TS_PACKAGES

    missing = [lang for lang in _TS_PACKAGES if _ts_parser(lang) is None]
    assert missing == [], f"tree-sitter grammars missing in CI: {missing}"
