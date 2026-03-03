# Configuration.md Updated ✅

## Changes Made

### 1. ✅ Fixed Line Number Reference
**Changed:** Removed specific line number `#L1171` from config.py link
**Reason:** Line numbers change as code evolves - better to link to file only

### 2. ✅ Added `chunk_external_properties` Parameter
**Added to main table:**
```markdown
| `chunk_external_properties` | Mapping of external property names to source metadata keys for chunk nodes | `None` | `CHUNK_EXTERNAL_PROPERTIES` |
```

**Added new section:** "Chunk External Properties" with:
- Usage example
- Explanation of key-value mapping
- Reserved keys warning (`text` and `chunkId`)
- Environment variable option

### 3. ✅ Updated `opensearch_engine` Description
**Changed:** `OpenSearch kNN engine` → `OpenSearch kNN engine (nmslib or faiss)`
**Reason:** Document both supported values

### 4. ✅ Completely Rewrote Nova 2 Section
**Old approach:** Auto-detection with string model names
**New approach:** Explicit import from `bedrock_utils`

**New content:**
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding

GraphRAGConfig.embed_model = Nova2MultimodalEmbedding('amazon.nova-2-multimodal-embeddings-v1:0')
GraphRAGConfig.embed_dimensions = 3072
```

**Removed:**
- Auto-detection explanation
- `is_nova_multimodal_embedding()` function reference
- JSON string configuration examples
- "Important: Set embed_dimensions BEFORE embed_model" warning (no longer needed)
- Reference to `bedrock_embedding.py` (file no longer exists)

**Added:**
- Clear explicit import pattern
- Customization example with all parameters
- Simplified API format explanation
- Supported dimensions note

---

## Summary of Updates

| Section | Status | Change |
|---------|--------|--------|
| Line number reference | ✅ Fixed | Removed specific line number |
| `chunk_external_properties` | ✅ Added | New parameter + section |
| `opensearch_engine` | ✅ Updated | Added `faiss` as valid value |
| Nova 2 Multimodal Embeddings | ✅ Rewritten | New explicit import pattern |

---

## What's Now Documented

### AWS's Features:
- ✅ `chunk_external_properties` - Map metadata to chunk properties
- ✅ `opensearch_engine` - Both `nmslib` and `faiss` options
- ✅ All existing parameters correctly documented

### Your Nova 2 Support:
- ✅ Explicit import from `bedrock_utils`
- ✅ Clear usage examples
- ✅ Customization options
- ✅ No confusing auto-detection magic

---

## Documentation is Now:
- ✅ AWS compliant
- ✅ Matches current implementation
- ✅ Clear and explicit (no auto-detection confusion)
- ✅ Complete with all parameters
- ✅ Ready for users!
