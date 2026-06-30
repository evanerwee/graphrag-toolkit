# codeproperty-graph

CPG domain layer for Joern-based code property graph analysis with delta ingestion.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              codeproperty-graph (domain)             │
│  CPGNode, CPGEdge, GraphDiff, DeltaIngestor         │
│  ManifestManager, tenant lifecycle                  │
├─────────────────────────────────────────────────────┤
│              document-graph (infra)                  │
│  Node, Edge, batch_nodes_unwind, batch_edges_unwind │
│  CypherBuilder, PipelineExecutor, multi-tenancy     │
├─────────────────────────────────────────────────────┤
│         graphrag-toolkit / lexical-graph (foundation)│
│  GraphStore, Neptune writer, AOSS writer            │
│  Lexical indexing, entity resolution, retrieval     │
└─────────────────────────────────────────────────────┘
```

This library sits on top of **document-graph** (typed property graph primitives) and
**lexical-graph** (graph storage foundation from AWS graphrag-toolkit). It adds
code-analysis-specific semantics:

- **CPG Models** — Joern node/edge types with `full_name`, `hash`, `signature`
- **GraphDiff** — compare two CPG states by method signature, return adds/removes/modified
- **ManifestManager** — S3-backed state tracking per repository
- **DeltaIngestor** — skip-or-replace orchestration with old tenant purge
- **Tenant Ops** — clean lifecycle management (`delete_tenant`)

## Usage

```python
from codeproperty_graph import DeltaIngestor

ingestor = DeltaIngestor(bucket="<your-artifacts-bucket>")

result = await ingestor.ingest(
    repo="amigo-core",
    job_id="abc-123",
    tenant_id="abc12345678901234567890",
    nodes_data=nodes,       # from Joern export
    edges_data=edges,       # from Joern export
    nodes_path="s3://bucket/cpg-exports/amigo-core/abc-123/nodes.json",
    edges_path="s3://bucket/cpg-exports/amigo-core/abc-123/edges.json",
    graph_store=neptune,
    write_fn=my_write_function,
)

# result: {"status": "SKIPPED", ...} or {"status": "INGESTED", "delta": "+2 -1 ~3 =150", ...}
```

## Delta Logic

1. Joern exports CPG → `nodes.json` + `edges.json`
2. Extract METHOD node signatures: `{full_name: hash}`
3. Compare against previous manifest in S3
4. If identical → **SKIP** (no Neptune writes)
5. If changed → **INGEST** full graph under new tenant, purge old tenant, update manifest

## Relationship to graphrag-toolkit

| Layer | Purpose | Example |
|-------|---------|---------|
| graphrag-toolkit | Graph storage, Neptune/AOSS writers | `GraphStore.execute_query()` |
| lexical-graph | Document chunking, entity extraction | `LexicalGraphIndex` |
| document-graph | Typed nodes/edges, Cypher generation | `batch_nodes_unwind()` |
| **codeproperty-graph** | CPG delta, manifests, Joern semantics | `DeltaIngestor.ingest()` |

## Install

```bash
pip install codeproperty-graph
```

## License

Apache-2.0
