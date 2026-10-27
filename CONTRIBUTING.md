# Contributing

1. Python 3.11+. `pip install -e '.[dev,local,lancedb,pdf]'`
2. Before a PR: `pytest && ruff check src tests && mypy src`
3. New formats need: a loader, tiny fixture in `tests/fixtures/`, golden
   chunk/locator tests, and a row in the README support matrix.
4. Non-obvious design choices need a short ADR in `docs/decisions/`.
5. Never stub silently — raise `NotImplementedError` with the phase it lands in.

By contributing you agree your work is licensed Apache-2.0.
