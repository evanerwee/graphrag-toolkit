# Document Graph Examples

Example notebooks for `graphrag-toolkit-document-graph` — structured data ingestion into Neptune.

## Prerequisites

- Python 3.10+
- `pip install graphrag-toolkit-document-graph` (base notebooks)
- `pip install graphrag-toolkit-document-graph[graphrag]` (Neptune notebooks)
- Amazon Neptune cluster (for Neptune notebooks)
- Amazon OpenSearch Serverless (for hybrid notebooks)

## Notebooks

### Getting Started

| # | Notebook | Neptune? | Description |
|---|----------|----------|-------------|
| 01 | Setup | Yes | Install packages, verify Neptune connection |
| 02 | Standalone ETL | **No** | Transform data + generate Cypher locally (no cloud needed) |
| 03 | Combined Extract-Load-Build | Yes | End-to-end: CSV → transform → Neptune |

### Pipeline Testing

| # | Notebook | Neptune? | Description |
|---|----------|----------|-------------|
| 04 | Full Pipeline Test | **No** | Test all pipeline stages locally (ingest → transform → construct) |
| 05a | Ingestors | **No** | Column select/rename/reorder, row filter, type conversion |
| 05b | Schema-Driven Pipeline | Yes | CSV → schema → full pipeline → Neptune |
| 05c | JSON-Driven Pipeline | Yes | JSON schema definition → automatic graph build |

### Transformers & Constructors

| # | Notebook | Neptune? | Description |
|---|----------|----------|-------------|
| 05d | Multi-Format Extraction | **No** | Parquet, Excel, XML, YAML extraction + schema discovery |
| 06a | Field Transformers | **No** | JSON flatten, UUID gen, regex, split, timestamp — 12 providers |
| 06b | Normalizers | **No** | Whitespace, nulls, case, enum, spelling, timestamp — 6 providers |
| 06c | Constructors | **No** | Node, edge, schema-driven, batch, dedup, 1:N, N:M — 8 patterns |
| 06d | Document Transformers | **No** | JSON→rows, text chunker, PII redactor |
| 06e | LLM Enrichers | Bedrock | AI-powered classification, entity extraction, language detection |
| 07 | Graph Write and Read | Yes | Direct Neptune read/write with batch UNWIND |
| 07b | Batch Performance | Optional | Benchmark: UNWIND vs individual MERGE (10-30x speedup) |
| 08 | Schema Providers | **No** | CSV, JSON, Parquet, Excel, S3, Glue, Static — discovery + validation |
| 09 | Transformer Deep Dive | **No** | Comprehensive guide to all 6 transformer categories |

### Hybrid Graph (document-graph + lexical-graph)

| # | Notebook | Neptune? | Description |
|---|----------|----------|-------------|
| 10a | Hybrid Data Processing | Yes | Structured data → Neptune → LlamaIndex Documents (with lineage) |
| 10b | Hybrid Lexical Indexing | Yes | Documents → lexical-graph → semantic search → lineage correlation |
| 11a | Mandela Hybrid Build | Yes | Real-world example: Wikipedia bio + structured events/orgs |
| 11b | Mandela Hybrid Query | Yes | Semantic + structured + hybrid correlation queries |
| 12 | Multi-Tenant Coexistence | Yes | Prove tenant isolation with shared Neptune cluster |

### Maintenance

| # | Notebook | Neptune? | Description |
|---|----------|----------|-------------|
| 99 | Cleanup | Yes | Remove old test tenants from Neptune |

## Data

Sample data in `data/` and `mandela_data/` for running examples without external dependencies.

## Running Locally (No Neptune)

Notebooks marked **No** in the Neptune column run entirely locally — no AWS credentials,
no Neptune cluster, no OpenSearch. They test pipeline logic, transformations, and Cypher generation.

Start with `02-Standalone-ETL.ipynb` for the zero-dependency introduction.
