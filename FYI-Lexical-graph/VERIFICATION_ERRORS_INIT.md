# Verification: errors.py & __init__.py ✅

## Status: Both Files Are Correct

### errors.py ✅
**Location:** `lexical-graph/src/graphrag_toolkit/lexical_graph/errors.py`

**Required for AWS compliance:**
- ✅ `ConfigurationError` - Used by `chunk_external_properties` validation in config.py

**Current content:**
```python
class ConfigurationError(Exception):
    pass

class ModelError(Exception):
    pass

class BatchJobError(Exception):
    pass

class IndexError(Exception):
    pass

class GraphQueryError(Exception):
    pass
```

**Status:** ✅ Matches AWS's version exactly

---

### __init__.py ✅
**Location:** `lexical-graph/src/graphrag_toolkit/lexical_graph/__init__.py`

**Required exports:**
- ✅ `ConfigurationError` - Exported so users can catch this exception

**Current exports:**
```python
from .errors import ModelError, BatchJobError, IndexError, GraphQueryError, ConfigurationError
```

**Status:** ✅ Matches AWS's version exactly

---

## Why ConfigurationError Is Needed

AWS added `ConfigurationError` for the `chunk_external_properties` feature in `config.py`:

```python
@chunk_external_properties.setter
def chunk_external_properties(self, chunk_external_properties: Optional[Dict[str, str]]) -> None:
    if chunk_external_properties and isinstance(chunk_external_properties, dict):
        if 'text' in chunk_external_properties:
            raise ConfigurationError("chunk_external_properties cannot contain a 'text' key")
        if 'chunkId' in chunk_external_properties:
            raise ConfigurationError("chunk_external_properties cannot contain a 'chunkId' key")
    self._chunk_external_properties = chunk_external_properties
```

This validates that users don't try to override reserved property names.

---

## Verification Complete

Both files are:
- ✅ AWS compliant
- ✅ Support the new `chunk_external_properties` feature
- ✅ Ready for your Nova 2 support (no conflicts)
- ✅ Match AWS's remote version exactly

No changes needed!
