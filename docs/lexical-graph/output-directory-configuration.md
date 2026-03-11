# Output Directory Configuration

This document describes the output directory configuration options available in the GraphRAG Toolkit.

## Overview

The GraphRAG Toolkit provides two new configuration options to control where files are written during processing:

- `local_output_dir`: Local staging directory for batch files
- `log_output_dir`: Directory for log files

These configurations are particularly useful for containerized deployments (EKS/Kubernetes) where the working directory may be read-only.

## Configuration Options

### local_output_dir

**Purpose**: Local output directory for batch staging files.

**Default**: `'output'`

**Description**: This directory is used by batch extractors to stage JSONL files before uploading to S3. For local development, the default `'output'` directory works well. For EKS/Kubernetes deployments, set to `'/tmp'` to use the writable temporary directory.

**Configuration Methods**:

1. **Environment Variable**:
   ```bash
   export LOCAL_OUTPUT_DIR="/tmp"
   ```

2. **Programmatic**:
   ```python
   from graphrag_toolkit.lexical_graph import GraphRAGConfig
   GraphRAGConfig.local_output_dir = '/tmp'
   ```

### log_output_dir

**Purpose**: Directory for log files.

**Default**: `None` (use filename as-is)

**Description**: When set, log filenames passed to `set_logging_config()` will be prefixed with this directory. When `None`, log filenames are used as provided. For EKS/Kubernetes deployments, set to `'/tmp'` to ensure logs are written to a writable location.

**Configuration Methods**:

1. **Environment Variable**:
   ```bash
   export LOG_OUTPUT_DIR="/tmp"
   ```

2. **Programmatic**:
   ```python
   from graphrag_toolkit.lexical_graph import GraphRAGConfig
   GraphRAGConfig.log_output_dir = '/tmp'
   ```

**Usage Example**:
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.logging import set_logging_config

# Set log directory
GraphRAGConfig.log_output_dir = '/tmp'

# This will create log file at /tmp/extraction.log
set_logging_config('INFO', filename='extraction.log')
```

## EKS/Kubernetes Deployment

For containerized deployments where the working directory is read-only, configure both directories to use `/tmp`:

**Via Environment Variables**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: graphrag-config
data:
  LOCAL_OUTPUT_DIR: "/tmp"
  LOG_OUTPUT_DIR: "/tmp"
```

**Via Code**:
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig

# Configure for container environment
GraphRAGConfig.local_output_dir = '/tmp'
GraphRAGConfig.log_output_dir = '/tmp'
```

## Implementation Details

### Timeout Configuration

The configuration also includes improved timeout settings for long-running extraction jobs:

- **LLM Operations**: 10 minutes read timeout, 2 minutes connect timeout
- **Embedding Operations**: 5 minutes read timeout, 2 minutes connect timeout
- **Retry Logic**: 10 attempts with adaptive mode

These settings help prevent SSL errors like "UNEXPECTED_EOF_WHILE_READING" that can occur during:
1. Long LLM operations
2. Istio sidecar interference with long-running connections  
3. Network interruptions during multi-hour extraction jobs

### Logging Integration

The `log_output_dir` setting integrates with the logging system:

- When `log_output_dir` is set and a relative filename is provided to `set_logging_config()`, the directory is automatically prefixed
- Absolute filenames are used as-is regardless of the `log_output_dir` setting
- The file handler is created dynamically only when a filename is provided, avoiding errors in read-only environments

## Migration Guide

### From Previous Versions

If you were previously hardcoding output paths:

**Before**:
```python
# Hardcoded paths
output_file = '/tmp/batch_output.jsonl'
log_file = '/tmp/extraction.log'
```

**After**:
```python
import os
from graphrag_toolkit.lexical_graph import GraphRAGConfig

# Configure directories
GraphRAGConfig.local_output_dir = '/tmp'
GraphRAGConfig.log_output_dir = '/tmp'

# Use relative paths - they'll be prefixed automatically
output_file = os.path.join(GraphRAGConfig.local_output_dir, 'batch_output.jsonl')
set_logging_config('INFO', filename='extraction.log')  # Creates /tmp/extraction.log
```

### Environment Variable Migration

Update your deployment configurations:

**Docker Compose**:
```yaml
services:
  graphrag:
    environment:
      - LOCAL_OUTPUT_DIR=/tmp
      - LOG_OUTPUT_DIR=/tmp
```

**Kubernetes**:
```yaml
spec:
  containers:
  - name: graphrag
    env:
    - name: LOCAL_OUTPUT_DIR
      value: "/tmp"
    - name: LOG_OUTPUT_DIR
      value: "/tmp"
```

## Testing

The configuration includes comprehensive tests covering:

- Default value behavior
- Environment variable reading
- Programmatic setting
- Value persistence
- Override behavior

Run tests with:
```bash
pytest lexical-graph/tests/graphrag_toolkit/lexical_graph/test_config_output_directories.py -v
```