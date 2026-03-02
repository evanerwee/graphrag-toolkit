# Documentation Review Findings — `lexical-graph`

> **Ground truth: the source code.** Everything below compares `README.md` against the actual implementation. Items are grouped by severity: **Critical** (code contradicts docs), **Gap** (significant undocumented behaviour), and **Minor** (style / polish).

---

## Summary

The README is intentionally brief — it is a landing page that points to `../docs/`. The critical issues are mostly in the code itself (docstring bugs, API inconsistencies) rather than in the README. The biggest gap is that none of the operational knobs (environment variables, prompt customisation, additional public API surface) are surfaced anywhere in the README.

---

## Critical — Code Contradicts Documentation

### 1. Method names: `extract_only` / `build_only` do not exist

The README and the class-level docstring of `LexicalGraphIndex` describe two methods — `extract_only` and `build_only`. The actual public methods are:

```python
# What the code has:
index.extract(nodes, ...)
index.build(nodes, ...)
index.extract_and_build(nodes, ...)
```

Anyone reading the docs and trying `.extract_only()` or `.build_only()` will get an `AttributeError`. The documentation must either rename these methods or the code must add aliases.

**Files:** [lexical_graph_index.py:421](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L421), [lexical_graph_index.py:475](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L475)

---

### 2. `IndexingConfig` docstring: chunk overlap is wrong

The docstring says:

> _"a default `SentenceSplitter` is used with a chunk size of 256 and an overlap of **20**"_

The code is:

```python
SentenceSplitter(chunk_size=256, chunk_overlap=25)  # overlap = 25, not 20
```

**File:** [lexical_graph_index.py:181](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L181)

---

### 3. `TenantId` docstring: max length is wrong

The class docstring says tenant IDs are _"between 1 to **10** characters"_. The validation code allows up to **25**:

```python
if len(value) > 25:   # actual limit
    return False
```

**File:** [tenant_id.py:47](src/graphrag_toolkit/lexical_graph/tenant_id.py#L47)

---

### 4. `to_indexing_config` is defined twice

The function is defined at line 190 and again at line 209 in the same file. The second definition (with the docstring) silently overwrites the first. This is dead code and could confuse anyone auditing the module.

**File:** [lexical_graph_index.py:190-250](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L190)

---

### 5. `VectorStoreFactory.for_composite` has a variable-name clash

```python
for k, v in v.indexes:   # outer loop variable 'v' is shadowed by inner 'v'
```

This loop re-assigns `v` inside itself, which will raise a `TypeError` on the second iteration (iterating over a non-iterable). The method is essentially broken.

**File:** [vector_store_factory.py:119](src/graphrag_toolkit/lexical_graph/storage/vector_store_factory.py#L119)

---

### 6. Async queries are silently unimplemented

`LexicalGraphQueryEngine` inherits from `BaseQueryEngine` and must implement `_aquery`. The current implementation is:

```python
async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
    pass   # returns None
```

There is no error, no warning, and no documentation saying async is unsupported. Callers using `await query_engine.aquery(...)` will silently receive `None`.

**File:** [lexical_graph_query_engine.py:563](src/graphrag_toolkit/lexical_graph/lexical_graph_query_engine.py#L563)

---

### 7. `for_traversal_based_search` and `for_semantic_guided_search` use different versioning parameter names

| Factory method | versioning param |
|---|---|
| `for_traversal_based_search` | `versioning` |
| `for_semantic_guided_search` | `enable_versioning` |

These should be identical since they are sibling factory methods on the same class.

**File:** [lexical_graph_query_engine.py:67-247](src/graphrag_toolkit/lexical_graph/lexical_graph_query_engine.py#L67)

---

## Gaps — Significant Undocumented Behaviour

### 8. Extraction always uses the default tenant, regardless of `tenant_id`

When `tenant_id` is set to a non-default value, the extraction phase (proposition + topic extraction) still writes output under the **default** tenant. Only the build phase uses the custom tenant. The code emits a warning, but this is a non-obvious semantic that affects anyone running multi-tenant indexing:

```python
if not self.tenant_id.is_default_tenant():
    logger.warning('TenantId has been set to non-default tenant id, but extraction will use default tenant id')
```

This behaviour should be clearly documented in the multi-tenancy guide.

**File:** [lexical_graph_index.py:445-455](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L445)

---

### 9. Context format defaults differ between factory methods and the constructor

| Entry point | `context_format` default |
|---|---|
| `for_traversal_based_search` | `'text'` |
| `for_semantic_guided_search` | always `'bedrock_xml'`, ignores kwarg |
| `__init__` direct | `'json'` |

The supported values (`'json'`, `'yaml'`, `'xml'`, `'text'`, `'bedrock_xml'`) are nowhere documented. `'bedrock_xml'` also automatically appends a `BedrockContextFormat` post-processor.

**File:** [lexical_graph_query_engine.py:132-244](src/graphrag_toolkit/lexical_graph/lexical_graph_query_engine.py#L132)

---

### 10. All configuration can be driven by environment variables

None of the following env vars appear in the README:

| Variable | Default | Purpose |
|---|---|---|
| `AWS_PROFILE` | — | AWS named profile |
| `AWS_REGION` | boto3 default | AWS region |
| `EXTRACTION_MODEL` | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | LLM for extraction |
| `RESPONSE_MODEL` | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | LLM for query responses |
| `EMBEDDINGS_MODEL` | `cohere.embed-english-v3` | Embedding model |
| `EMBEDDINGS_DIMENSIONS` | `1024` | Embedding vector size |
| `EXTRACTION_NUM_WORKERS` | `2` | Worker processes for extraction |
| `EXTRACTION_BATCH_SIZE` | `4` | Docs per batch |
| `EXTRACTION_NUM_THREADS_PER_WORKER` | `4` | Threads inside each worker |
| `BUILD_NUM_WORKERS` | `2` | Worker processes for build |
| `BUILD_BATCH_SIZE` | `4` | Items per build batch |
| `BUILD_BATCH_WRITE_SIZE` | `25` | Items per graph write call |
| `BATCH_WRITES_ENABLED` | `true` | Enable/disable batch graph writes |
| `INCLUDE_DOMAIN_LABELS` | `false` | Add domain labels to graph nodes |
| `INCLUDE_LOCAL_ENTITIES` | `false` | Include local-context entities |
| `INCLUDE_CLASSIFICATION_IN_ENTITY_ID` | `true` | Include classification in entity hash |
| `ENABLE_CACHE` | `false` | LLM response caching |
| `RERANKING_MODEL` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Local reranker |
| `BEDROCK_RERANKING_MODEL` | `cohere.rerank-v3-5:0` | Bedrock reranker |
| `OPENSEARCH_ENGINE` | `nmslib` | OpenSearch kNN engine |
| `ENABLE_VERSIONING` | `false` | Enable versioned updates |

**File:** [config.py](src/graphrag_toolkit/lexical_graph/config.py)

---

### 11. Connection string formats for stores are not documented

The README shows one example (`neptune-db://…`, `aoss://…`) but never explains the full set of accepted prefixes:

| Store | Connection string prefix |
|---|---|
| Neptune Analytics | `neptune-graph://[graph-id]` |
| Neptune Database | `neptune-db://[hostname]` or any hostname ending `.neptune.amazonaws.com` |
| Neo4j | `bolt://`, `bolt+ssc://`, `bolt+s://`, `neo4j://`, `neo4j+ssc://`, `neo4j+s://` |
| OpenSearch Serverless | `aoss://[url]` |
| pgvector | resolved via `PGVectorIndexFactory` |
| S3 Vectors | resolved via `S3VectorIndexFactory` |
| Dummy (no-op) | `None` / any unrecognised string → `DummyGraphStore`/`DummyVectorIndex` |

**Files:** [neptune_graph_stores.py:22-24](src/graphrag_toolkit/lexical_graph/storage/graph/neptune_graph_stores.py#L22), [neo4j_graph_store_factory.py:8](src/graphrag_toolkit/lexical_graph/storage/graph/neo4j_graph_store_factory.py#L8)

---

### 12. Prompt customisation system is completely undocumented

The query engine supports four prompt provider backends that control the system and user prompts sent to the LLM:

| Provider | Trigger | Config class |
|---|---|---|
| `StaticPromptProvider` | default (no config needed) | `StaticPromptProviderConfig` |
| `FilePromptProvider` | env var `PROMPT_PATH` | `FilePromptProviderConfig` |
| `S3PromptProvider` | env var `PROMPT_S3_BUCKET` | `S3PromptProviderConfig` |
| `BedrockPromptProvider` | env vars `SYSTEM_PROMPT_ARN` / `USER_PROMPT_ARN` | `BedrockPromptProviderConfig` |

The `PromptProviderFactory` auto-detects which to use based on which env vars are set. Custom prompts can also be passed directly to `LexicalGraphQueryEngine.__init__` via `system_prompt` and `user_prompt` parameters, or via a `prompt_provider` kwarg.

**Files:** [prompts/](src/graphrag_toolkit/lexical_graph/prompts/)

---

### 13. Undocumented public API on `LexicalGraphIndex`

Beyond `extract_and_build`, the index class has:

- **`get_stats()`** — returns a dict with node counts (`source`, `chunk`, `topic`, `statement`, `fact`, `entity`) and two connectivity metrics (`localConnectivity`, `globalConnectivity`).
- **`get_sources(...)`** — queries the graph for source document metadata, supports filtering by `source_id`, list of IDs, `FilterConfig`, or dict, plus versioning and ordering.
- **`delete_sources(...)`** — same filter API as `get_sources`, deletes matching sources from both the graph store and the vector store.

**File:** [lexical_graph_index.py:596-785](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L596)

---

### 14. `add_versioning_info()` utility is a public API

This function is exported from `__init__.py` but is never described:

```python
from graphrag_toolkit.lexical_graph import add_versioning_info

metadata = add_versioning_info(
    metadata={},
    id_fields=['url'],     # fields that determine document identity across versions
    valid_from=1234567890  # unix timestamp (ms) when this version became valid
)
```

It adds the internal versioning keys to a document's metadata dict before indexing, enabling point-in-time querying.

**File:** [versioning.py:35-44](src/graphrag_toolkit/lexical_graph/versioning.py#L35)

---

### 15. `BatchConfig` is undocumented for all required fields

The README mentions batch extraction in passing but never shows a `BatchConfig` construction. The required parameters are non-obvious AWS infrastructure values:

```python
BatchConfig(
    role_arn='arn:aws:iam::…',     # IAM role for Bedrock batch jobs
    region='us-east-1',
    bucket_name='my-bucket',        # S3 bucket for job I/O
    key_prefix='batch/',            # Optional S3 prefix
    s3_encryption_key_id=None,      # Optional KMS key
    subnet_ids=[],                  # VPC subnet IDs
    security_group_ids=[],          # VPC security groups
    max_batch_size=25000,
    max_num_concurrent_batches=3,
    delete_on_success=True
)
```

**File:** [extract/batch_config.py](src/graphrag_toolkit/lexical_graph/indexing/extract/batch_config.py)

---

### 16. Built-in document readers are not documented

The `indexing/load/` sub-package includes pre-built reader providers for many source types that can be composed via `FileBasedDocs` and `S3BasedDocs`. None appear in the README:

PDF, Advanced PDF, DOCX, PPTX, CSV, JSON, Markdown, Web pages, Wikipedia, YouTube transcripts, GitHub repositories, Database tables, S3 directories, and a general Directory reader.

**Directory:** [indexing/load/readers/providers/](src/graphrag_toolkit/lexical_graph/indexing/load/readers/providers/)

---

### 17. Asyncio patching is a silent side effect of importing the module

On import, `__init__.py` patches `llama_index.core.async_utils.asyncio_run` to support Jupyter notebooks. This is done unconditionally and silently. It can interact unexpectedly with other code using LlamaIndex in the same process.

**File:** [__init__.py:16-38](src/graphrag_toolkit/lexical_graph/__init__.py#L16)

---

### 18. `ExtractionConfig.preferred_topics` is missing from the class docstring

The docstring lists `preferred_entity_classifications` and `infer_entity_classifications` but omits `preferred_topics`, which accepts the same type (`PREFERRED_VALUES_PROVIDER_TYPE` — a list or a callable provider) and seeds the LLM with preferred topic names during extraction.

Similarly, `BuildConfig.enable_versioning` is present in `__init__` but absent from both the class docstring and the attribute list.

**File:** [lexical_graph_index.py:44-131](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L44)

---

### 19. `GraphRAGConfig` is a singleton instance, not a class to instantiate

The README and configuration docs may imply users should create a `GraphRAGConfig()` object. It is actually a module-level singleton:

```python
GraphRAGConfig = _GraphRAGConfig()   # single shared instance
```

Usage is always mutation of this singleton:

```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
GraphRAGConfig.aws_region = 'eu-west-1'
GraphRAGConfig.extraction_llm = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
```

Setting `aws_profile` or `aws_region` automatically clears all cached boto3 clients.

**File:** [config.py:1171](src/graphrag_toolkit/lexical_graph/config.py#L1171)

---

### 20. README installation version is behind the codebase

The README install URL pins `v3.15.5`:

```
pip install https://github.com/awslabs/graphrag-toolkit/archive/refs/tags/v3.15.5.zip#...
```

`pyproject.toml` shows version `3.16.2-SNAPSHOT`. The README should either always reference `main` or be updated on each release.

---

### 21. `ResilientClient` and SSO login flow are undocumented

`GraphRAGConfig` wraps all boto3 clients in `ResilientClient`, which retries on `ExpiredToken` / `RequestExpired` / `InvalidClientTokenId` errors by refreshing the client. When an SSO profile is detected, it validates the token age (1 hour) and will automatically run `aws sso login` if the token is stale. This is significant behaviour for users running in SSO environments or long-running jobs.

**File:** [config.py:94-229](src/graphrag_toolkit/lexical_graph/config.py#L94)

---

## Minor — Polish / Clarity

### 22. Querying example in README does not use a context manager

The indexing example wraps stores in `with ... as store:`, the querying example does not. This inconsistency could mislead users into thinking store connections are not closeable resources during query time. Either both examples should use context managers, or the README should note when it is and is not required.

### 23. `_get_prompts`, `_get_prompt_modules`, `_update_prompts` are stub implementations

These three methods from the LlamaIndex `PromptMixin` interface all `pass` (return `None`). They will silently fail if any LlamaIndex tooling tries to inspect or modify prompts on the query engine.

**File:** [lexical_graph_query_engine.py:566-573](src/graphrag_toolkit/lexical_graph/lexical_graph_query_engine.py#L566)

### 24. `prompt_provider_config.py` contains a "Suggested Next Enhancements" comment block

A 10-item enhancement wishlist is embedded in a production source file as a comment block. This should be either moved to an issue tracker or removed.

**File:** [prompts/prompt_provider_config.py:237-282](src/graphrag_toolkit/lexical_graph/prompts/prompt_provider_config.py#L237)

### 25. `LexicalGraphQueryEngine.verbose` controls answer verbosity

The `verbose` kwarg (default `True`) is passed to the LLM prompt as `answer_mode='fully' if self.verbose else 'concisely'`. This is not documented. It only affects the non-streaming code path.

**File:** [lexical_graph_query_engine.py:356-361](src/graphrag_toolkit/lexical_graph/lexical_graph_query_engine.py#L356)

### 26. `InferClassifications` runs as a pre-processor before chunking/extraction

When `infer_entity_classifications=True` in `ExtractionConfig`, an `InferClassifications` step is added as a **pre-processor** (not a pipeline component). It samples documents and uses an LLM to infer entity classification names before the main extraction loop begins. This means an extra LLM round-trip happens once per batch, not per document.

**File:** [lexical_graph_index.py:365-388](src/graphrag_toolkit/lexical_graph/lexical_graph_index.py#L365)

### 27. S3 Vectors additional dependency is undocumented in README

The README lists installation extras for OpenSearch, pgvector, and Neo4j. There is no equivalent section for S3 Vectors even though an `S3VectorIndexFactory` and `s3_vector_indexes.py` exist.

---

## Recommended Actions

| Priority | Action |
|---|---|
| Critical | Rename `extract_only`/`build_only` in docs to `extract`/`build` (or add aliases to code) |
| Critical | Fix `to_indexing_config` duplicate definition |
| Critical | Fix `VectorStoreFactory.for_composite` variable shadowing bug |
| Critical | Document or implement `_aquery` |
| High | Align `versioning` / `enable_versioning` parameter names across factory methods |
| High | Fix `TenantId` and `IndexingConfig` docstring length/overlap bugs |
| High | Add environment variable reference table to configuration docs |
| High | Document connection string formats for all stores |
| High | Document prompt customisation system |
| High | Document `BatchConfig` required fields with an example |
| Medium | Document `get_stats`, `get_sources`, `delete_sources` API |
| Medium | Document `add_versioning_info` utility |
| Medium | Clarify tenant-extraction behaviour (always default tenant) |
| Medium | Update install URL to current release |
| Medium | Document `context_format` options and their defaults per factory |
| Low | Move `prompt_provider_config.py` enhancement notes out of source |
| Low | Document `verbose` kwarg on query engine |
| Low | Add S3 Vectors to the "Additional dependencies" section of README |
