# Missing from Your config.py

Your `config.py` is missing AWS's `chunk_external_properties` feature added after your Nova 2 work.

---

## 1. Missing Import (Line ~22)

Add after the botocore imports:

```python
from graphrag_toolkit.lexical_graph.errors import ConfigurationError
```

---

## 2. Missing Default Constant (Line ~54)

Add after `DEFAULT_ENABLE_VERSIONING = False`:

```python
DEFAULT_CHUNK_EXTERNAL_PROPERTIES = None
```

---

## 3. Missing Class Attribute (Line ~293)

Add in the `_GraphRAGConfig` dataclass attributes section:

```python
_chunk_external_properties: Optional[Dict[str, str]] = None
```

---

## 4. Missing Property Implementation (Lines ~1175-1226)

Add at the end of the class, before `GraphRAGConfig = _GraphRAGConfig()`:

```python
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
    
    Example:
        {
            'article_code': 'article_id',      # chunk.article_code from metadata['article_id']
            'document_type': 'doc_type',       # chunk.document_type from metadata['doc_type']
            'department': 'dept_code'          # chunk.department from metadata['dept_code']
        }
    
    Returns:
        Optional[Dict[str, str]]: Dictionary mapping chunk property names to metadata keys,
            or None if not configured.
    """
    if self._chunk_external_properties is None:
        env_value = os.environ.get('CHUNK_EXTERNAL_PROPERTIES', DEFAULT_CHUNK_EXTERNAL_PROPERTIES)
        if env_value and _is_json_string(env_value):
            self._chunk_external_properties = json.loads(env_value)
        else:
            self._chunk_external_properties = env_value
    return self._chunk_external_properties

@chunk_external_properties.setter
def chunk_external_properties(self, chunk_external_properties: Optional[Dict[str, str]]) -> None:
    """
    Sets the mapping of external property names to source metadata keys.
    
    Args:
        chunk_external_properties: Dictionary mapping chunk property names to metadata keys,
            or None to disable the feature.
            
    Example:
        GraphRAGConfig.chunk_external_properties = {
            'article_code': 'article_id',
            'document_type': 'doc_type'
        }
    """
    if chunk_external_properties and isinstance(chunk_external_properties, dict):
        if 'text' in chunk_external_properties:
            raise ConfigurationError("chunk_external_properties cannot contain a 'text' key")
        if 'chunkId' in chunk_external_properties:
            raise ConfigurationError("chunk_external_properties cannot contain a 'chunkId' key")
    self._chunk_external_properties = chunk_external_properties
```

---

## Summary

**4 additions needed:**
1. ✅ Import `ConfigurationError`
2. ✅ Add `DEFAULT_CHUNK_EXTERNAL_PROPERTIES = None`
3. ✅ Add `_chunk_external_properties` class attribute
4. ✅ Add `chunk_external_properties` property getter and setter

This is AWS's feature for mapping external metadata properties to chunk nodes, unrelated to your Nova 2 work.
