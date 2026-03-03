# Configuration Documentation Gaps

Analysis of `docs/lexical-graph/configuration.md` vs `lexical-graph/src/graphrag_toolkit/lexical_graph/config.py`

---

## Issues Found

### 1. ❌ **Missing: `chunk_external_properties` Parameter**

**Status:** Completely undocumented

**What it is:** New configuration parameter added after the last doc update that allows mapping external property names to source metadata keys for chunk nodes.

**Source code:**
```python
DEFAULT_CHUNK_EXTERNAL_PROPERTIES = None

@property
def chunk_external_properties(self) -> Optional[Dict[str, str]]:
    """
    Gets the mapping of external property names to source metadata keys.
    
    This property allows you to configure which metadata fields from source documents
    should be extracted and added as properties on chunk nodes in the graph database.
    This enables querying and filtering chunks by business-specific identifiers.
    
    The mapping is a dictionary where:
    - Key: The property name to use on the chunk node (e.g., 'article_code', 'document_id')
    - Value: The metadata key to extract from source document (e.g., 'article_id', 'doc_ref')
    """
```

**Environment variable:** `CHUNK_EXTERNAL_PROPERTIES` (likely)

**Recommendation:** Add to main configuration table with description and example usage.

---

### 2. ⚠️ **Incomplete: `opensearch_engine` Valid Values**

**Status:** Partially documented

**Issue:** Documentation shows `opensearch_engine` parameter but doesn't list all valid values.

**Documentation says:**
```markdown
| `opensearch_engine` | OpenSearch kNN engine | `nmslib` | `OPENSEARCH_ENGINE` |
```

**Source code shows:**
```python
DEFAULT_OPENSEARCH_ENGINE = 'nmslib'
```

**Valid values:** `'nmslib'` (default), `'faiss'` (added in commit 1713ab17)

**Recommendation:** Update description to: `OpenSearch kNN engine (nmslib or faiss)`

---

### 3. ⚠️ **Incomplete: Nova 2 Embedding `embed_purpose` Default**

**Status:** Documented but with wrong default value

**Documentation says:**
```markdown
| `embed_purpose` | Embedding optimization purpose | `TEXT_RETRIEVAL` | ...
```

**Source code shows:**
```python
# In config.py embed_model setter:
embed_purpose=config.get('embed_purpose', 'RETRIEVAL'),  # Default is 'RETRIEVAL', not 'TEXT_RETRIEVAL'
```

**Issue:** The default value in the code is `'RETRIEVAL'` but documentation shows `'TEXT_RETRIEVAL'`

**Recommendation:** Verify correct default and update documentation accordingly.

---

### 4. ✅ **Correct: Line Number Reference**

**Status:** Needs update

**Documentation says:**
```markdown
It is created once at import time ([`config.py`](../../lexical-graph/src/graphrag_toolkit/lexical_graph/config.py#L1171))
```

**Source code:** The singleton is created at the last line of config.py, which is now beyond line 1171 due to new additions.

**Current line:** Approximately line 1230 (after `chunk_external_properties` addition)

**Recommendation:** Update line reference or remove specific line number (just link to file).

---

### 5. ❌ **Missing: Nova 2 LLM Support**

**Status:** Completely undocumented

**What it is:** Support for Amazon Nova 2 series LLM models using `DirectBedrockLLM` instead of `BedrockConverse`.

**Source code shows:** The `_to_llm()` method only uses `BedrockConverse`, no Nova 2 LLM support visible in current code.

**Note:** This was mentioned in the previous analysis but appears to have been removed or not yet merged into main branch.

**Recommendation:** If Nova 2 LLM support exists in another branch, document it. Otherwise, ignore for now.

---

### 6. ✅ **Correct: All Other Parameters**

The following parameters are correctly documented:
- ✅ `extraction_llm` - correct default and description
- ✅ `response_llm` - correct default and description  
- ✅ `embed_model` - correct default and description
- ✅ `embed_dimensions` - correct default (1024)
- ✅ `extraction_num_workers` - correct default (2)
- ✅ `extraction_num_threads_per_worker` - correct default (4)
- ✅ `extraction_batch_size` - correct default (4)
- ✅ `build_num_workers` - correct default (2)
- ✅ `build_batch_size` - correct default (4)
- ✅ `build_batch_write_size` - correct default (25)
- ✅ `batch_writes_enabled` - correct default (True)
- ✅ `include_domain_labels` - correct default (False)
- ✅ `include_local_entities` - correct default (False)
- ✅ `include_classification_in_entity_id` - correct default (True)
- ✅ `enable_versioning` - correct default (False)
- ✅ `enable_cache` - correct default (False)
- ✅ `reranking_model` - correct default
- ✅ `bedrock_reranking_model` - correct default
- ✅ `aws_profile` - correctly documented
- ✅ `aws_region` - correctly documented

---

## Summary Table

| Issue | Severity | Action Required |
|-------|----------|-----------------|
| Missing `chunk_external_properties` | High | Add new parameter to configuration table with description and example |
| `opensearch_engine` incomplete | Medium | Add `faiss` as valid value in description |
| Nova 2 `embed_purpose` default wrong | Low | Verify and correct default value (`RETRIEVAL` vs `TEXT_RETRIEVAL`) |
| Line number reference outdated | Low | Update or remove specific line number in link |
| Nova 2 LLM support | N/A | Not in current main branch - ignore for now |

---

## Recommended Documentation Updates

### 1. Add `chunk_external_properties` to main table

Insert after `enable_versioning`:

```markdown
| `chunk_external_properties` | Mapping of external property names to source metadata keys for chunk nodes | `None` | `CHUNK_EXTERNAL_PROPERTIES` |
```

Add usage example in a new section:

```markdown
#### Chunk External Properties

You can configure which metadata fields from source documents should be extracted and added as properties on chunk nodes:

\`\`\`python
GraphRAGConfig.chunk_external_properties = {
    'article_code': 'article_id',
    'document_type': 'doc_type'
}
\`\`\`

This enables querying chunks by business-specific identifiers in your graph database.
```

### 2. Update `opensearch_engine` description

Change:
```markdown
| `opensearch_engine` | OpenSearch kNN engine | `nmslib` | `OPENSEARCH_ENGINE` |
```

To:
```markdown
| `opensearch_engine` | OpenSearch kNN engine (`nmslib` or `faiss`) | `nmslib` | `OPENSEARCH_ENGINE` |
```

### 3. Fix Nova 2 `embed_purpose` default

In the Nova 2 Multimodal Embeddings section, change:

```markdown
| `embed_purpose` | Embedding optimization purpose | `TEXT_RETRIEVAL` | ...
```

To:
```markdown
| `embed_purpose` | Embedding optimization purpose | `RETRIEVAL` | ...
```

### 4. Update line number reference

Change:
```markdown
It is created once at import time ([`config.py`](../../lexical-graph/src/graphrag_toolkit/lexical_graph/config.py#L1171))
```

To:
```markdown
It is created once at import time ([`config.py`](../../lexical-graph/src/graphrag_toolkit/lexical_graph/config.py))
```

(Remove specific line number to avoid future maintenance)
