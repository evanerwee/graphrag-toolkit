# Reader Provider Enhancements

## Overview
This document describes the enhancements made to the lexical-graph reader providers, focusing on improved error handling, comprehensive logging, and the addition of a new universal directory reader provider.

## Key Enhancements

### 1. Enhanced Error Handling

All reader providers now implement robust error handling patterns:

#### Import Error Handling
- **Clear dependency messages**: When required dependencies are missing, providers now show specific installation commands
- **Graceful degradation**: Import failures are caught and re-raised with helpful context

```python
try:
    from llama_index.readers.file.pymu_pdf import PyMuPDFReader
except ImportError as e:
    logger.error("Failed to import PyMuPDFReader: missing pymupdf")
    raise ImportError(
        "PyMuPDFReader requires 'pymupdf'. Install with: pip install pymupdf"
    ) from e
```

#### Runtime Error Handling
- **Input validation**: All providers validate input sources before processing
- **Exception chaining**: Original exceptions are preserved using `from e` syntax
- **Contextual error messages**: Errors include specific information about what failed

```python
if not input_source:
    logger.error("No input source provided to PDFReaderProvider")
    raise ValueError("input_source cannot be None or empty")

try:
    documents = self._reader.load_data(file_path=processed_paths[0])
except Exception as e:
    logger.error(f"Failed to read PDF from {input_source}: {e}", exc_info=True)
    raise RuntimeError(f"Failed to read PDF: {e}") from e
```

### 2. Comprehensive Logging

#### Structured Logging Pattern
All providers follow a consistent logging pattern:

- **Debug level**: Initialization and configuration details
- **Info level**: Operation progress and success metrics
- **Error level**: Failures with full exception context

```python
logger = logging.getLogger(__name__)

# Initialization
logger.debug(f"Initialized PDFReaderProvider with return_full_document={config.return_full_document}")

# Progress tracking
logger.info(f"Reading PDF from: {input_source}")
logger.info(f"Successfully read {len(documents)} document(s) from PDF")

# Error reporting
logger.error(f"Failed to read PDF from {input_source}: {e}", exc_info=True)
```

#### Log Context
- **File paths**: Original and processed paths are logged
- **Document counts**: Success metrics include document counts
- **Configuration**: Key configuration parameters are logged at debug level
- **Exception traces**: Full stack traces are captured with `exc_info=True`

### 3. Universal Directory Reader Provider

#### Purpose
The new `UniversalDirectoryReaderProvider` provides a unified interface for reading from both local directories and S3-based document collections.

#### Key Features

**Dual Mode Operation**:
- **Local mode**: Uses LlamaIndex's `SimpleDirectoryReader` for local file system access
- **S3 mode**: Uses GraphRAG's `S3BasedDocs` for S3-based document collections

**Intelligent Source Detection**:
```python
def read(self, input_source: Optional[Union[str, Dict[str, str]]] = None) -> List[Document]:
    """Read from local or S3 based on config/input."""
    
    if isinstance(input_source, dict) or self.config.bucket_name:
        return self._read_from_s3(input_source)
    else:
        return self._read_from_local(input_source)
```

**Comprehensive Configuration**:
```python
class UniversalDirectoryReaderConfig(ReaderProviderConfig):
    # Local directory options
    input_dir: Optional[str] = None
    input_files: Optional[List[str]] = None
    exclude_hidden: bool = True
    recursive: bool = False
    required_exts: Optional[List[str]] = None
    file_extractor: Optional[Dict[str, Any]] = None
    metadata_fn: Optional[callable] = None
    
    # S3BasedDocs options
    region: Optional[str] = None
    bucket_name: Optional[str] = None
    key_prefix: Optional[str] = None
    collection_id: Optional[str] = None
```

#### Usage Examples

**Local Directory Reading**:
```python
config = UniversalDirectoryReaderConfig(
    input_dir="/path/to/documents",
    recursive=True,
    required_exts=[".pdf", ".txt"],
    metadata_fn=lambda path: {"source": "local", "path": path}
)
reader = UniversalDirectoryReaderProvider(config)
documents = reader.read()
```

**S3 Collection Reading**:
```python
config = UniversalDirectoryReaderConfig(
    region="us-east-1",
    bucket_name="my-documents",
    key_prefix="collections",
    collection_id="project-docs",
    metadata_fn=lambda path: {"source": "s3", "path": path}
)
reader = UniversalDirectoryReaderProvider(config)
documents = reader.read()
```

**Dynamic Source Selection**:
```python
# Local reading
documents = reader.read("/local/path")

# S3 reading via dict
s3_config = {
    "region": "us-west-2",
    "bucket_name": "docs-bucket",
    "key_prefix": "data",
    "collection_id": "analysis"
}
documents = reader.read(s3_config)
```

## Error Handling Patterns

### 1. Validation Errors
```python
if not all([region, bucket_name, key_prefix, collection_id]):
    logger.error("Missing S3 configuration")
    raise ValueError("S3 requires: region, bucket_name, key_prefix, collection_id")
```

### 2. Import Errors
```python
try:
    from graphrag_toolkit.lexical_graph.indexing.load import S3BasedDocs
except ImportError as e:
    logger.error("Failed to import S3BasedDocs")
    raise ImportError("S3BasedDocs not available") from e
```

### 3. Runtime Errors
```python
try:
    documents = reader.load_data()
    logger.info(f"Successfully read {len(documents)} document(s) from local")
except Exception as e:
    logger.error(f"Failed to read from local: {e}", exc_info=True)
    raise RuntimeError(f"Failed to read documents: {e}") from e
```

## Logging Best Practices

### 1. Consistent Logger Naming
```python
logger = logging.getLogger(__name__)
```

### 2. Appropriate Log Levels
- **DEBUG**: Configuration details, internal state
- **INFO**: Operation progress, success metrics
- **ERROR**: Failures with context

### 3. Exception Context
```python
logger.error(f"Failed to read from S3: {e}", exc_info=True)
```

### 4. Structured Messages
```python
logger.info(f"Reading from S3: s3://{bucket_name}/{key_prefix}/{collection_id}")
```

## Benefits

### 1. Improved Debugging
- Clear error messages with specific failure context
- Full exception traces for troubleshooting
- Progress tracking through structured logging

### 2. Better User Experience
- Helpful dependency installation messages
- Clear validation error descriptions
- Consistent error handling across all providers

### 3. Enhanced Flexibility
- Universal directory reader supports both local and S3 sources
- Dynamic source detection based on input type
- Unified configuration interface

### 4. Production Readiness
- Robust error handling prevents crashes
- Comprehensive logging aids monitoring
- Graceful degradation when dependencies are missing

## Migration Notes

### Existing Code Compatibility
All existing reader providers maintain backward compatibility while adding enhanced error handling and logging.

### New Universal Provider
The `UniversalDirectoryReaderProvider` can replace separate local and S3 directory readers in many use cases, providing a single interface for both scenarios.

### Logging Configuration
Applications should configure logging appropriately to capture the enhanced log output:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Future Enhancements

1. **Metrics Integration**: Add performance metrics to logging output
2. **Retry Logic**: Implement automatic retry for transient failures
3. **Async Support**: Add asynchronous reading capabilities
4. **Caching**: Implement document caching for frequently accessed sources