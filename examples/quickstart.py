"""Quickstart: embed a directory into LanceDB, then query it in a later run.

    pip install 'embedify[local,lancedb,pdf]'
    python examples/quickstart.py
"""

import embedify

result = embedify.embed("docs/", sink=embedify.LanceDBSink("./embedify.lancedb", "chunks"))
print(f"embedded {result.stats.units_embedded} units, errors: {result.errors}")

hits = embedify.query("how do refunds work?", db="./embedify.lancedb")
for record, distance in hits:
    print(distance, record.source_uri, record.locator, record.text_preview[:120])
