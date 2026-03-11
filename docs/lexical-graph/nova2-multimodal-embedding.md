# Nova 2 Multimodal Embedding Support

This document describes how to use Amazon Nova 2 multimodal embeddings with the GraphRAG Toolkit.

## Overview

Amazon Nova 2 multimodal embeddings use a different API format than standard Bedrock embedding models. The `Nova2MultimodalEmbedding` class provides a custom implementation that handles this API format while maintaining compatibility with the LlamaIndex embedding interface.

## Usage

### Basic Usage

```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding

# Create Nova 2 embedding instance
GraphRAGConfig.embed_model = Nova2MultimodalEmbedding('amazon.nova-2-multimodal-embeddings-v1:0')
GraphRAGConfig.embed_dimensions = 3072
```

### Advanced Configuration

```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding

# Create with custom configuration
nova_embedding = Nova2MultimodalEmbedding(
    model_name='amazon.nova-2-multimodal-embeddings-v1:0',
    embed_dimensions=3072,  # 1024 or 3072
    embed_purpose='TEXT_RETRIEVAL',  # TEXT_RETRIEVAL, GENERIC_RETRIEVAL, CLASSIFICATION, CLUSTERING
    truncation_mode='END'  # END or NONE
)

GraphRAGConfig.embed_model = nova_embedding
GraphRAGConfig.embed_dimensions = 3072
```

## Configuration Parameters

### embed_dimensions
- **Type**: `int`
- **Default**: `3072`
- **Options**: `1024` or `3072`
- **Description**: The dimensionality of the embedding vectors

### embed_purpose
- **Type**: `str`
- **Default**: `"TEXT_RETRIEVAL"`
- **Options**: 
  - `"TEXT_RETRIEVAL"`: Optimized for text retrieval tasks
  - `"GENERIC_RETRIEVAL"`: General-purpose retrieval
  - `"CLASSIFICATION"`: Optimized for classification tasks
  - `"CLUSTERING"`: Optimized for clustering tasks
- **Description**: The intended use case for the embeddings

### truncation_mode
- **Type**: `str`
- **Default**: `"END"`
- **Options**: `"END"` or `"NONE"`
- **Description**: How to handle text that exceeds the model's maximum input length
  - `"END"`: Truncate text at the end
  - `"NONE"`: Do not truncate (may cause errors for long text)

## API Format

Nova 2 multimodal embeddings use a specific API request format:

```json
{
    "taskType": "SINGLE_EMBEDDING",
    "singleEmbeddingParams": {
        "embeddingDimension": 3072,
        "embeddingPurpose": "TEXT_RETRIEVAL",
        "text": {
            "truncationMode": "END",
            "value": "text to embed"
        }
    }
}
```

The response format is:

```json
{
    "embeddings": [
        {
            "embeddingType": "TEXT",
            "embedding": [0.1, 0.2, 0.3, ...]
        }
    ]
}
```

## Error Handling and Retry Logic

The `Nova2MultimodalEmbedding` class includes robust error handling and retry logic:

### Retryable Errors
The following errors are automatically retried with exponential backoff:
- `ModelErrorException`
- `ThrottlingException`
- `ServiceUnavailableException`
- `InternalServerException`
- `ServiceException`
- Errors containing phrases like "unexpected error", "try your request again", "service unavailable", "throttl"

### Retry Configuration
- **Maximum Retries**: 5 attempts
- **Base Delay**: 1.0 seconds
- **Maximum Delay**: 30.0 seconds
- **Backoff Strategy**: Exponential with jitter

### Input Validation
The class validates input text and raises `ValueError` for:
- Empty strings
- Whitespace-only strings

## Multiprocessing Support

The `Nova2MultimodalEmbedding` class supports multiprocessing through custom pickle methods:

```python
import pickle
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding

# Create embedding instance
embedding = Nova2MultimodalEmbedding('amazon.nova-2-multimodal-embeddings-v1:0')

# Pickle and unpickle (for multiprocessing)
pickled_data = pickle.dumps(embedding)
restored_embedding = pickle.loads(pickled_data)

# The bedrock client is automatically recreated when needed
```

## Integration with GraphRAG Toolkit

### Document Processing

```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding
from graphrag_toolkit.lexical_graph.indexing.build import LexicalGraphBuilder

# Configure Nova 2 embeddings
GraphRAGConfig.embed_model = Nova2MultimodalEmbedding('amazon.nova-2-multimodal-embeddings-v1:0')
GraphRAGConfig.embed_dimensions = 3072

# Build lexical graph with Nova 2 embeddings
builder = LexicalGraphBuilder(GraphRAGConfig)
graph = builder.build_graph_from_documents(documents)
```

### Batch Processing

The embedding class works seamlessly with batch processing operations:

```python
from graphrag_toolkit.lexical_graph.indexing.extract import BatchLLMPropositionExtractor

# Configure extraction with Nova 2 embeddings
extractor = BatchLLMPropositionExtractor()
propositions = extractor.extract_propositions(documents)
```

## Performance Considerations

### Embedding Dimensions
- **1024 dimensions**: Faster processing, lower memory usage, slightly reduced accuracy
- **3072 dimensions**: Higher accuracy, slower processing, higher memory usage

### Batch Size Optimization
For large-scale processing, consider adjusting batch sizes based on your embedding dimensions:

```python
# For 3072-dimensional embeddings, use smaller batches
GraphRAGConfig.extraction_batch_size = 2
GraphRAGConfig.build_batch_size = 2

# For 1024-dimensional embeddings, larger batches are feasible
GraphRAGConfig.extraction_batch_size = 4
GraphRAGConfig.build_batch_size = 4
```

## Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```
   ResourceNotFoundException: Model not found
   ```
   - Ensure you have access to Nova 2 models in your AWS region
   - Check that the model ID is correct: `amazon.nova-2-multimodal-embeddings-v1:0`

2. **Throttling Errors**
   ```
   ThrottlingException: Rate limit exceeded
   ```
   - The class automatically retries with exponential backoff
   - Consider reducing batch sizes or adding delays between requests

3. **Empty Text Errors**
   ```
   ValueError: Text cannot be empty or whitespace-only
   ```
   - Ensure your documents contain non-empty text content
   - Check for documents with only whitespace or special characters

### Debugging

Enable debug logging to see detailed information about embedding operations:

```python
import logging

# Enable debug logging for Nova 2 embeddings
logging.getLogger('graphrag_toolkit.lexical_graph.utils.bedrock_utils').setLevel(logging.DEBUG)
```

## Migration from Standard Bedrock Embeddings

If you're migrating from standard Bedrock embeddings to Nova 2:

### Before (Standard Bedrock)
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig

GraphRAGConfig.embed_model = "cohere.embed-english-v3"
GraphRAGConfig.embed_dimensions = 1024
```

### After (Nova 2)
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding

GraphRAGConfig.embed_model = Nova2MultimodalEmbedding('amazon.nova-2-multimodal-embeddings-v1:0')
GraphRAGConfig.embed_dimensions = 3072
```

### Considerations
- Nova 2 embeddings have different dimensions (1024 or 3072 vs. 1024 for Cohere)
- Nova 2 may have different performance characteristics
- Existing vector indices may need to be rebuilt with new embeddings

## AWS IAM Permissions

Ensure your AWS role/user has the necessary permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/amazon.nova-*"
            ]
        }
    ]
}
```

## Regional Availability

Nova 2 models may not be available in all AWS regions. Check the AWS Bedrock documentation for current regional availability.

## Cost Considerations

Nova 2 multimodal embeddings may have different pricing than standard text embeddings. Consult the AWS Bedrock pricing page for current rates.