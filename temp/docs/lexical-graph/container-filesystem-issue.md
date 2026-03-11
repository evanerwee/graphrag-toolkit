# Container Filesystem Issue - Hardcoded 'output' Directory

## Issue Summary

**Severity**: High  
**Affects**: All containerized deployments (EKS, ECS, Docker with read-only filesystems)  
**Symptom**: `PermissionError: [Errno 13] Permission denied: 'output'`

## Root Cause

Multiple files in graphrag-toolkit have hardcoded `'output'` as the default directory for staging batch files and temporary outputs. In containerized environments where the working directory is read-only (standard practice for EKS/Kubernetes), this causes extraction and build pipelines to fail immediately.

## Affected Files

The following files had hardcoded `output_dir='output'` or `batch_inference_dir='output'` defaults:

| File | Location | Issue |
|------|----------|-------|
| `batch_extractor_base.py` | `indexing/extract/` | `batch_inference_dir='output'` parameter |
| `batch_llm_proposition_extractor.py` | `indexing/extract/` | `batch_inference_dir='output'` parameter |
| `batch_llm_proposition_extractor_sync.py` | `indexing/extract/` | `batch_inference_dir='output'` parameter |
| `batch_topic_extractor.py` | `indexing/extract/` | `batch_inference_dir='output'` parameter |
| `batch_topic_extractor_sync.py` | `indexing/extract/` | `batch_inference_dir='output'` parameter |
| `file_system_tap.py` | `indexing/extract/` | `output_dir='output'` parameter |
| `checkpoint.py` | `indexing/build/` | `output_dir='output'` parameter |
| `bedrock_knowledge_base.py` | `indexing/load/` | `output_dir='output'` parameter |
| `logging.py` | root module | Log file paths not configurable |

## The Fix

### 1. Added Configuration Properties to GraphRAGConfig

```python
# In config.py
DEFAULT_LOCAL_OUTPUT_DIR = 'output'  # Local staging directory for batch files
DEFAULT_LOG_OUTPUT_DIR = None  # Log file directory (None = use filename as-is)

@property
def local_output_dir(self) -> str:
    """Local output directory for batch staging files."""
    if self._local_output_dir is None:
        self._local_output_dir = os.environ.get('LOCAL_OUTPUT_DIR', DEFAULT_LOCAL_OUTPUT_DIR)
    return self._local_output_dir

@property
def log_output_dir(self) -> Optional[str]:
    """Directory for log files."""
    if self._log_output_dir is None:
        self._log_output_dir = os.environ.get('LOG_OUTPUT_DIR', DEFAULT_LOG_OUTPUT_DIR)
    return self._log_output_dir
```

### 2. Updated All Affected Files

Changed from hardcoded defaults to using `GraphRAGConfig.local_output_dir`:

```python
# Before (broken in containers)
def __init__(self, checkpoint_name, output_dir='output', enabled=True):
    self.checkpoint_dir = self.prepare_output_directories(checkpoint_name, output_dir)

# After (container-compatible)
def __init__(self, checkpoint_name, output_dir=None, enabled=True):
    from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
    resolved_output_dir = output_dir if output_dir is not None else GraphRAGConfig.local_output_dir
    self.checkpoint_dir = self.prepare_output_directories(checkpoint_name, resolved_output_dir)
```

## Usage

### For Container/EKS Deployments

Set environment variables:
```bash
export LOCAL_OUTPUT_DIR=/tmp
export LOG_OUTPUT_DIR=/tmp
```

Or configure programmatically:
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig

GraphRAGConfig.local_output_dir = '/tmp'
GraphRAGConfig.log_output_dir = '/tmp'
```

### For Local Development

No changes needed - defaults to `'output'` directory as before.

## Design Recommendation

For future development, avoid hardcoding filesystem paths in function/method signatures. Instead:

1. Use `None` as the default parameter value
2. Resolve the actual path at runtime from `GraphRAGConfig`
3. This allows environment-specific configuration without code changes

```python
# ✅ Good pattern
def __init__(self, output_dir=None):
    from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
    self.output_dir = output_dir or GraphRAGConfig.local_output_dir

# ❌ Bad pattern
def __init__(self, output_dir='output'):
    self.output_dir = output_dir
```

## Testing

Verified fix by running extraction workflow in EKS:
- Pod: `extraction-7e26c09d-20260308-8ntl2`
- Document: NIST SP 800-53 catalog (~3110 chunks)
- Result: Successfully processing without permission errors

## Related Documentation

- [Configuration Guide](./configuration.md#container-and-eks-deployment)
