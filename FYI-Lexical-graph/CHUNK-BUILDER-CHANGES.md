# Chunk Builder Changes - NOT Nova 2 Related

## Answer: NO - Not Related to Nova 2

The changes in `chunk_node_builder.py` and `chunk_graph_builder.py` are for **AWS's `chunk_external_properties` feature**, not Nova 2.

---

## What These Files Do

### chunk_node_builder.py
**Purpose:** Builds chunk nodes with metadata

**AWS's chunk_external_properties implementation:**
```python
# Add external properties if configured
external_props = GraphRAGConfig.chunk_external_properties
if external_props and isinstance(external_props, dict):
    valid_source_metadata = metadata['source']['metadata']
    chunk_metadata = metadata['chunk']['metadata']
    for prop_name, metadata_key in external_props.items():
        if metadata_key in valid_source_metadata:
            chunk_metadata[prop_name] = valid_source_metadata[metadata_key]
```

**What it does:** Maps source document metadata to chunk node properties based on user configuration.

### chunk_graph_builder.py
**Purpose:** Writes chunk nodes to the graph database

**AWS's chunk_external_properties implementation:**
```python
# Add external properties if present
for key, value in chunk_metadata.get('metadata', {}).items():
    chunk_property_setters.append(f'chunk.{key} = params.{key}')
    properties_c[key] = value
```

**What it does:** Adds the external properties as actual properties on the chunk node in the graph.

---

## Why These Changes Matter

**Feature:** `chunk_external_properties` allows users to:
```python
GraphRAGConfig.chunk_external_properties = {
    'article_code': 'article_id',
    'document_type': 'doc_type'
}
```

**Result:** Chunk nodes in the graph get these properties, enabling queries like:
```cypher
MATCH (c:__Chunk__ {article_code: 'ABC123'})
RETURN c
```

---

## Status

✅ **RESTORED** - Both files now have AWS's chunk_external_properties implementation
❌ **NOT Nova 2 related** - These are for metadata mapping feature
✅ **AWS compliant** - Matches AWS's implementation exactly

---

## Summary

- **Nova 2 changes:** Only in `bedrock_utils.py` and `config.py`
- **Chunk builder changes:** For AWS's `chunk_external_properties` feature
- **Both restored:** Files now match AWS's implementation
