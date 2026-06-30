# READMORE: Domain Graphs and the DRY Principle in graphrag-toolkit

## The GraphRAG Pattern

[GraphRAG](https://graphrag.com/) establishes a taxonomy of graph types for retrieval-augmented generation:

| Graph Type | Input | Structure | Query Style |
|------------|-------|-----------|-------------|
| **Entity Graph** | Unstructured text | LLM-extracted entities + relationships | Semantic search, traversal |
| **Domain Graph** | Structured/domain-specific data | Typed nodes with schema | Cypher, exact match, domain queries |
| **Lexical Graph** | Text chunks | Chunk → entity → topic hierarchy | Hybrid (vector + graph) |

graphrag-toolkit implements all three — and proves they compose into a single Neptune store:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Amazon Neptune                               │
├─────────────────────────────────────────────────────────────────┤
│  lexical-graph         │  document-graph    │  codeproperty-graph│
│  (Entity/Lexical)      │  (Domain Graph)    │  (Domain Graph)    │
│                        │                    │                    │
│  Text → Entities       │  CSV/JSON → Nodes  │  Code → CPG Nodes  │
│  Chunks → Embeddings   │  Schema → Edges    │  Joern → Delta     │
│                        │                    │                    │
│  "What is X?"          │  "Find all Y"      │  "What changed?"   │
│  (semantic)            │  (structured)      │  (code analysis)   │
└─────────────────────────────────────────────────────────────────┘
```

## document-graph: A Domain Graph for Structured Data

`document-graph` is a **Domain Graph** in the GraphRAG taxonomy. It takes structured data (CSV, JSON, Parquet, Excel, XML, YAML) and builds a typed property graph with declared schema.

What makes it a domain graph (vs. an entity graph):
- **Schema-first**: You declare node types, edge types, and property mappings
- **Typed nodes**: Each node has a specific label (`User`, `Account`, `Event`) — not generic "Entity"
- **Deterministic construction**: Same input → same graph (no LLM extraction variance)
- **Structured queries**: Cypher with known schema, not semantic similarity

What makes it part of the toolkit (not standalone):
- Uses lexical-graph's `GraphStoreFactory` for Neptune connectivity
- Uses lexical-graph's `TenantId.format_label()` for multi-tenancy
- Shares the same Neptune cluster — hybrid queries span both graph types
- Follows the same namespace, packaging, and distribution pattern

## codeproperty-graph: Domain Graph + DRY in Action

`codeproperty-graph` is both a **Domain Graph** (code analysis domain) and the clearest demonstration of the **DRY principle** in this toolkit.

### The Domain

Code Property Graphs (CPGs) are the standard representation for static code analysis. Joern produces them from source code in 9 languages. codeproperty-graph ingests them into Neptune for:
- Call graph analysis ("what calls what?")
- Dependency tracking ("what changed between builds?")
- Security analysis ("trace data flow from input to output")
- Architecture visualization ("show me the class hierarchy")

### The DRY Architecture

codeproperty-graph is **~200 lines of domain-specific code**. Everything else is reused:

```
codeproperty-graph (domain layer — what's NEW)
├── schema.py          — Joern type enums, property specs
├── models.py          — CPGNode, CPGEdge, Manifest (with from_joern())
├── graph_diff.py      — Method signature comparison
├── manifest_manager.py — S3 state tracking
├── delta_ingestor.py  — Skip-or-replace orchestration
└── tenant_ops.py      — Tenant lifecycle

document-graph (infrastructure — REUSED, not duplicated)
├── Node, Edge models
├── batch_nodes_unwind(), batch_edges_unwind()
├── CypherBuilder (tenant-scoped label generation)
└── Multi-tenancy (TenantId.format_label())

lexical-graph (foundation — REUSED, not duplicated)
├── GraphStoreFactory (Neptune connection)
├── VectorStoreFactory (OpenSearch connection)
├── TenantId (shared tenant format)
└── Graph write infrastructure
```

What codeproperty-graph does NOT contain:
- ❌ No Cypher generation code (uses document-graph's `batch_nodes_unwind`)
- ❌ No Neptune connection code (uses lexical-graph's `GraphStoreFactory`)
- ❌ No tenant label formatting (uses lexical-graph's `TenantId`)
- ❌ No graph write batching logic (uses document-graph's UNWIND patterns)
- ❌ No multi-tenancy implementation (inherits from the shared format)

**This is what DRY looks like at the package level**: a domain-specific layer that adds *only* domain semantics (CPG types, delta logic, manifests) without reimplementing any infrastructure.

### Why This Matters

1. **New domain graphs are cheap to build.** If you have a new domain (security findings, infrastructure state, compliance records), you write ~200 lines of domain models + delta logic. The graph infra already exists.

2. **All domain graphs are compatible.** Because they share tenant format and graph store, you can query across domains in a single Cypher traversal. A security domain graph can link to the code property graph via shared identifiers.

3. **Infrastructure improvements benefit all domains.** When document-graph gets batch performance improvements or lexical-graph gets a new graph store backend, codeproperty-graph inherits them automatically — zero changes required.

4. **The pattern is provably correct.** codeproperty-graph's 17 tests pass against the shared infrastructure. If the contract (TenantId format, Node/Edge models, GraphStore interface) holds, any domain graph built on it will work.

## The Layered Architecture

```
┌─────────────────────────────────────────────────────┐
│         Domain Graphs (your code)                    │
│  codeproperty-graph  │  your-domain-graph           │
│  ~200 lines          │  ~200 lines                  │
├─────────────────────────────────────────────────────┤
│         document-graph (typed graph infra)            │
│  Node/Edge models, CypherBuilder, batch writes,      │
│  schema providers, transformers, constructors        │
├─────────────────────────────────────────────────────┤
│         lexical-graph (foundation)                    │
│  GraphStore, VectorStore, TenantId, Neptune/AOSS     │
│  connection, entity extraction, semantic indexing    │
└─────────────────────────────────────────────────────┘
```

Each layer has a clear responsibility:
- **lexical-graph**: How to connect to and write to graph/vector stores
- **document-graph**: How to transform structured data into typed graphs
- **domain graphs**: What the domain-specific types and logic are

## Building Your Own Domain Graph

To add a new domain (e.g., infrastructure state, security findings, compliance records):

```python
# 1. Define your domain models (~50 lines)
@dataclass
class SecurityFinding:
    id: str
    severity: str
    resource_arn: str
    ...

# 2. Define your delta logic (~50 lines)
class FindingsDiff:
    @staticmethod
    def compare(prev, curr): ...

# 3. Define your ingestor (~100 lines)
class FindingsIngestor:
    async def ingest(self, findings, graph_store, tenant_id): ...
        # Uses document-graph's batch_nodes_unwind
        # Uses lexical-graph's GraphStore
        # Uses shared TenantId format

# That's it. ~200 lines. Full Neptune integration.
```

The toolkit pattern means you focus on *domain semantics* — what makes your domain unique — and inherit all the graph infrastructure for free.

## Summary

| Package | GraphRAG Type | Lines of Domain Code | Infrastructure Reused |
|---------|--------------|---------------------|-----------------------|
| lexical-graph | Entity + Lexical Graph | Foundation | — |
| document-graph | Domain Graph (generic) | ~5000 | lexical-graph |
| codeproperty-graph | Domain Graph (code) | ~200 | document-graph + lexical-graph |
| *your-domain-graph* | Domain Graph (yours) | ~200 | document-graph + lexical-graph |

The architecture proves that GraphRAG domain graphs are composable, reusable, and cheap to build when the infrastructure layer is solid.
