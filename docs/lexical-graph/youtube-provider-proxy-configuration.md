# YouTube Reader Provider Proxy Configuration

This document describes how to configure proxy settings for the YouTube Reader Provider to bypass cloud provider IP blocks.

## Overview

YouTube blocks requests from cloud provider IP addresses (AWS, GCP, Azure). When running the GraphRAG Toolkit on cloud infrastructure, you may encounter errors when trying to fetch YouTube transcripts. The YouTube Reader Provider supports proxy configuration to bypass these restrictions.

## Problem

When running on cloud infrastructure, you may see errors like:
```
TranscriptsDisabled: The uploader has not made transcripts available for this video
```

This often occurs not because transcripts are disabled, but because YouTube blocks requests from cloud provider IP ranges.

## Solution

Configure a proxy server that YouTube doesn't block. The YouTube Reader Provider supports proxy configuration at the provider level.

## Configuration Methods

### 1. Provider Configuration (Recommended)

Configure the proxy directly in the YouTubeReaderConfig:

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import YouTubeReaderConfig
from graphrag_toolkit.lexical_graph.indexing.load.readers.providers.youtube_reader_provider import YouTubeReaderProvider

# Configure with proxy
config = YouTubeReaderConfig(
    language="en",
    proxy_url="http://username:password@proxy.example.com:8080"
)

provider = YouTubeReaderProvider(config)
documents = provider.read("https://www.youtube.com/watch?v=VIDEO_ID")
```

### 2. Environment Variable

Set the `YOUTUBE_PROXY_URL` environment variable:

```bash
export YOUTUBE_PROXY_URL="http://username:password@proxy.example.com:8080"
```

Then use the provider without explicit proxy configuration:

```python
config = YouTubeReaderConfig(language="en")
provider = YouTubeReaderProvider(config)
documents = provider.read("https://www.youtube.com/watch?v=VIDEO_ID")
```

### 3. Priority Order

The provider uses the following priority order for proxy configuration:

1. **Provider config** (`config.proxy_url`) - highest priority
2. **Environment variable** (`YOUTUBE_PROXY_URL`) - fallback
3. **No proxy** - if neither is configured

## Proxy URL Format

The proxy URL should follow the standard format:

```
http://[username:password@]host:port
https://[username:password@]host:port
```

Examples:
- `http://proxy.example.com:8080` (no authentication)
- `http://user:pass@proxy.example.com:8080` (with authentication)
- `https://secure.proxy.com:3128` (HTTPS proxy)

## Implementation Details

### Environment Variable Handling

The provider temporarily sets `HTTP_PROXY` and `HTTPS_PROXY` environment variables during YouTube API calls:

```python
# Before API call
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# Make YouTube API call
transcript = api.fetch(video_id)

# After API call - restore original values
# (automatic cleanup in finally block)
```

### Security Considerations

1. **Credential Masking**: Proxy credentials are masked in log output
2. **Environment Restoration**: Original proxy environment variables are always restored
3. **Exception Safety**: Proxy cleanup occurs even if API calls fail

### Logging

The provider logs proxy configuration at appropriate levels:

```python
# Info level - proxy configured (credentials masked)
logger.info("YouTube proxy configured: proxy.example.com:8080")

# Debug level - no proxy
logger.debug("No YouTube proxy configured - may fail on cloud provider IPs")
```

## Usage Examples

### Basic Usage with Proxy

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import YouTubeReaderConfig
from graphrag_toolkit.lexical_graph.indexing.load.readers.providers.youtube_reader_provider import YouTubeReaderProvider

# Configure YouTube reader with proxy
config = YouTubeReaderConfig(
    language="en",
    proxy_url="http://proxy.company.com:8080"
)

provider = YouTubeReaderProvider(config)

# Read single video
documents = provider.read("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Read multiple videos
video_urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=oHg5SJYRHA0"
]
documents = provider.read(video_urls)
```

### Environment Variable Configuration

```bash
# Set environment variable
export YOUTUBE_PROXY_URL="http://proxy.company.com:8080"

# Run your application
python your_script.py
```

```python
# No explicit proxy in config - uses environment variable
config = YouTubeReaderConfig(language="en")
provider = YouTubeReaderProvider(config)
documents = provider.read("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

### Docker/Kubernetes Deployment

**Docker Compose:**
```yaml
services:
  graphrag:
    image: your-graphrag-image
    environment:
      - YOUTUBE_PROXY_URL=http://proxy.company.com:8080
```

**Kubernetes:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: graphrag-config
data:
  YOUTUBE_PROXY_URL: "http://proxy.company.com:8080"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphrag
spec:
  template:
    spec:
      containers:
      - name: graphrag
        envFrom:
        - configMapRef:
            name: graphrag-config
```

### With Authentication

```python
# Proxy with username/password authentication
config = YouTubeReaderConfig(
    language="en",
    proxy_url="http://myuser:mypassword@secure-proxy.com:3128"
)

provider = YouTubeReaderProvider(config)
documents = provider.read("https://www.youtube.com/watch?v=VIDEO_ID")
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   ConnectionError: HTTPSConnectionPool(host='www.youtube.com', port=443)
   ```
   - Verify proxy URL is correct
   - Check proxy server is accessible from your network

2. **Authentication Failed**
   ```
   ProxyError: 407 Proxy Authentication Required
   ```
   - Verify username/password in proxy URL
   - Check proxy authentication method

3. **Still Getting Blocked**
   ```
   TranscriptsDisabled: The uploader has not made transcripts available
   ```
   - Verify proxy is not using cloud provider IPs
   - Try a different proxy server
   - Check if the video actually has transcripts available

### Testing Proxy Configuration

Test your proxy configuration manually:

```bash
# Test proxy connectivity
curl -x http://proxy.example.com:8080 https://www.youtube.com/

# Test with authentication
curl -x http://user:pass@proxy.example.com:8080 https://www.youtube.com/
```

### Debug Logging

Enable debug logging to see proxy configuration details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your YouTube reader code here
```

## Migration from Global Configuration

If you were previously using a global `youtube_proxy_url` configuration, migrate to provider-level configuration:

**Before (deprecated):**
```python
from graphrag_toolkit.lexical_graph import GraphRAGConfig

# This approach is deprecated
GraphRAGConfig.youtube_proxy_url = "http://proxy.example.com:8080"
```

**After (recommended):**
```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import YouTubeReaderConfig

# Provider-specific configuration
config = YouTubeReaderConfig(
    language="en",
    proxy_url="http://proxy.example.com:8080"
)
```

## Best Practices

1. **Use Environment Variables**: For production deployments, use environment variables instead of hardcoding proxy URLs
2. **Secure Credentials**: Store proxy credentials securely (e.g., AWS Secrets Manager, Kubernetes Secrets)
3. **Test Connectivity**: Always test proxy connectivity before deploying
4. **Monitor Logs**: Monitor logs for proxy-related errors and connection issues
5. **Fallback Strategy**: Consider implementing fallback logic for when proxies are unavailable

## Security Notes

- Proxy credentials are automatically masked in log output
- Original environment variables are always restored after API calls
- Consider using HTTPS proxies for additional security
- Regularly rotate proxy credentials if using authentication