[[Home](./)]

## Streaming Responses

This document covers true token-by-token streaming with graphrag-toolkit, following patterns from [Pydantic AI Chat App](https://ai.pydantic.dev/examples/chat-app/) and [FastAPI Chatbot Best Practices](https://dev.to/vipascal99/building-a-full-stack-ai-chatbot-with-fastapi-backend-and-react-frontend-51ph).

### Topics

- [AI4Triage Implementation](#ai4triage-implementation)
- [Why True Streaming Matters](#why-true-streaming-matters)
- [Enabling Streaming](#enabling-streaming)
- [FastAPI SSE Pattern](#fastapi-sse-pattern)
- [Chat Application Pattern](#chat-application-pattern)
- [Frontend Integration](#frontend-integration)
- [Common Anti-Patterns](#common-anti-patterns)

### AI4Triage Implementation

#### Changes Made (March 2026)

We added true token streaming support to the chat feature. These changes are **backwards compatible** - existing functionality is unchanged.

#### Architecture Overview

```
UI (Streamlit) → Control-Plane (SSE Passthrough) → Data-Plane (True Streaming) → Bedrock LLM
     ↑                    ↑                              ↑
  api_client.py    factory_proxy_handler.py      query_service.py
  (stream client)  (SSE passthrough)             (execute_query_streaming)
```

#### Files Changed

| File | Change |
|------|--------|
| `data-plane/.../services/query_service.py` | New `execute_query_streaming()` method |
| `data-plane/.../api/endpoints.py` | Updated `lexical_graph_stream_chat_message` to use streaming |
| `control-plane/.../factory_proxy_handler.py` | Added SSE passthrough for streaming endpoints |
| `ui-layer/app/api_client.py` | Updated `stream_chat_message` to handle true SSE |

---

##### 1. New Method: `QueryService.execute_query_streaming()`

**File:** `data-plane/src/ai4triage/data_plane/integrations/lexical_graph/services/query_service.py`

```python
async def execute_query_streaming(
    self,
    query: str,
    config: Optional[QueryEngineConfig] = None,
    graph_store_url: Optional[str] = None,
    vector_store_url: Optional[str] = None,
) -> AsyncGenerator[Tuple[str, List[Dict[str, Any]], Dict[str, Any]], None]:
```

**Purpose:** Provides true token-by-token streaming from the LLM, yielding tokens as they arrive from Bedrock.

**Yields:**
- During streaming: `(token_text, [], {})` - each token as it arrives
- Final yield: `("", source_nodes, metadata)` - sources and timing info after completion

**Backwards Compatibility:**
- This is a NEW method - `execute_query()` is unchanged
- Use `execute_query()` for non-streaming (waits for full response)
- Use `execute_query_streaming()` for true token streaming

---

##### 2. Updated Endpoint: `lexical_graph_stream_chat_message`

**File:** `data-plane/src/ai4triage/data_plane/integrations/lexical_graph/api/endpoints.py`

**Before (fake streaming):**
```python
# Called non-streaming execute_query, then split words
query_result = await query_service.execute_query(query, config)
response_text = query_result.get("response", "")
words = response_text.split()
for word in words:
    yield f"event: token\ndata: {json.dumps({'text': word})}\n\n"
    await asyncio.sleep(0.01)  # Fake delay
```

**After (true streaming):**
```python
# Uses new streaming method - tokens arrive as LLM generates them
async for token, sources, meta in query_service.execute_query_streaming(query, config):
    if token:
        yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
```

**Message Storage Order:**
- User message stored BEFORE streaming (we know the input)
- Assistant message stored AFTER streaming completes (need full response for storage and moderation)

---

##### 3. Control-Plane SSE Passthrough

**File:** `control-plane/src/ai4triage/control_plane/decision_router/api/handlers/factory_proxy_handler.py`

The control-plane now detects streaming endpoints (by checking for "stream" in the path/operation_id) and uses `httpx.stream()` to pass SSE events through without buffering.

**Before:** Control-plane buffered entire response, wrapped in JSON
**After:** Control-plane streams SSE events directly to client

```python
# Detect streaming endpoint
is_streaming_endpoint = "stream" in operation_path.lower() or "stream" in operation_id_snake_case.lower()

if is_streaming_endpoint:
    # Use httpx.stream() for SSE passthrough
    async with client.stream(...) as response:
        async for chunk in response.aiter_text():
            yield chunk
```

---

##### 4. UI Client Update

**File:** `infrastructure/platform/docker/ui-layer/app/api_client.py`

The UI client now uses `client.stream()` to receive SSE events as they arrive, with fallback to JSON parsing for backwards compatibility.

```python
async with client.stream("POST", url, ...) as response:
    if "text/event-stream" in content_type:
        # TRUE STREAMING: SSE passthrough
        async for line in response.aiter_lines():
            yield line
    else:
        # FALLBACK: JSON response (old behavior)
        response_json = json.loads(await response.aread())
        ...
```

---

##### SSE Event Format

The streaming endpoint emits these Server-Sent Events:

| Event | Data | When |
|-------|------|------|
| `token` | `{"text": "...", "index": N}` | Each token as it arrives |
| `source` | `{"source": "...", "topic": "...", "score": N}` | After streaming, for each source |
| `moderation` | `{"categories_flagged": [...], "action_taken": "..."}` | If output flagged |
| `metadata` | `{"total_ms": N, "retrieve_ms": N, ...}` | Timing info |
| `done` | `{"status": "complete", "correlation_id": "..."}` | Stream complete |
| `error` | `{"code": "...", "message": "..."}` | On error |

### Why True Streaming Matters

True streaming provides immediate feedback to users as the LLM generates tokens. This creates a responsive "typing" effect similar to ChatGPT, rather than waiting for the entire response before displaying anything.

**Benefits:**
- Perceived latency drops from seconds to milliseconds
- Users can read the response as it's generated
- Early cancellation is possible if the response is going in the wrong direction
- Better UX for long responses

### Enabling Streaming

Pass `streaming=True` when creating the query engine:

```python
from graphrag_toolkit.lexical_graph import LexicalGraphQueryEngine
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory, VectorStoreFactory

with (
    GraphStoreFactory.for_graph_store(graph_store_url) as graph_store,
    VectorStoreFactory.for_vector_store(vector_store_url) as vector_store,
):
    query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
        graph_store,
        vector_store,
        streaming=True,  # Enable streaming
        tenant_id=tenant_id,
    )
    
    response = query_engine.query(query)
    
    # response.response_gen is a generator that yields tokens
    if response.response_gen:
        for token in response.response_gen:
            print(token, end="", flush=True)
```

When `streaming=True`, the response object contains:
- `response_gen`: A generator yielding tokens as they arrive from the LLM
- `source_nodes`: Retrieved source documents (available immediately)
- `metadata`: Timing and context information

### FastAPI SSE Pattern

Server-Sent Events (SSE) is the recommended pattern for streaming responses to web clients. This follows the pattern from the Pydantic AI chat app example.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    """Stream chat response using SSE."""
    
    async def stream_generator():
        # Yield user message immediately for instant feedback
        yield f"event: user\ndata: {json.dumps({'content': request.message})}\n\n"
        
        with (
            GraphStoreFactory.for_graph_store(graph_url) as graph_store,
            VectorStoreFactory.for_vector_store(vector_url) as vector_store,
        ):
            query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
                graph_store, vector_store, streaming=True
            )
            
            response = query_engine.query(request.message)
            
            # Stream tokens as they arrive
            if response.response_gen:
                for token in response.response_gen:
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
            
            # Stream source nodes after response completes
            for node in response.source_nodes:
                yield f"event: source\ndata: {json.dumps({'source': node.text})}\n\n"
            
            # Signal completion
            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### Chat Application Pattern

For chat applications with conversation history, follow this pattern inspired by the Pydantic AI example:

```python
from datetime import datetime, timezone

@app.post("/chat/")
async def post_chat(prompt: str, db: Database) -> StreamingResponse:
    """Chat endpoint with history and streaming."""
    
    async def stream_messages():
        # 1. Immediately yield user message for instant UI feedback
        yield json.dumps({
            'role': 'user',
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'content': prompt,
        }).encode() + b'\n'
        
        # 2. Get conversation history for context
        messages = await db.get_messages()
        
        # 3. Build query with history context
        history_context = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in messages[-10:]  # Last 10 messages
        ])
        query_with_context = f"Previous conversation:\n{history_context}\n\nUser: {prompt}"
        
        # 4. Stream the response
        with (
            GraphStoreFactory.for_graph_store(graph_url) as graph_store,
            VectorStoreFactory.for_vector_store(vector_url) as vector_store,
        ):
            query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
                graph_store, vector_store, streaming=True
            )
            
            response = query_engine.query(query_with_context)
            full_response = ""
            
            if response.response_gen:
                for token in response.response_gen:
                    full_response += token
                    yield json.dumps({
                        'role': 'assistant',
                        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                        'content': token,
                        'partial': True,
                    }).encode() + b'\n'
        
        # 5. Store messages AFTER streaming completes
        await db.add_message('user', prompt)
        await db.add_message('assistant', full_response)
    
    return StreamingResponse(stream_messages(), media_type='text/plain')
```

**Key Points:**
1. Yield user message immediately for instant feedback
2. Load conversation history before querying
3. Stream tokens as they arrive
4. Store messages only after streaming completes (you need the full response)

### Frontend Integration

**JavaScript/React EventSource pattern:**

```javascript
async function streamChat(message) {
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep incomplete event in buffer
        
        for (const line of lines) {
            if (line.startsWith('event: token')) {
                const data = JSON.parse(line.split('data: ')[1]);
                appendToken(data.text); // Update UI incrementally
            } else if (line.startsWith('event: done')) {
                onComplete();
            }
        }
    }
}
```

**Python Streamlit pattern:**

```python
import streamlit as st
import requests

def stream_response(message):
    """Stream response in Streamlit."""
    response = requests.post(
        f"{API_URL}/chat/stream",
        json={"message": message},
        stream=True,
    )
    
    placeholder = st.empty()
    full_response = ""
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data:'):
                data = json.loads(line[5:])
                if 'text' in data:
                    full_response += data['text']
                    placeholder.markdown(full_response + "▌")
    
    placeholder.markdown(full_response)
    return full_response
```

### Common Anti-Patterns

#### ❌ WRONG: Fake Streaming (Word Splitting)

```python
# DON'T DO THIS - This is NOT true streaming
response = query_engine.query(query)  # Waits for full response
full_text = str(response.response)

# Fake "streaming" by splitting words
for word in full_text.split():
    yield f"event: token\ndata: {json.dumps({'text': word})}\n\n"
    await asyncio.sleep(0.01)  # Artificial delay
```

This defeats the purpose of streaming because:
- User waits for the entire response before seeing anything
- The artificial delay adds latency, not reduces it
- No early cancellation benefit

#### ✅ CORRECT: True Token Streaming

```python
# DO THIS - True streaming from LLM
response = query_engine.query(query)  # Returns immediately with generator

if response.response_gen:
    for token in response.response_gen:  # Yields as LLM generates
        yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
```

#### ❌ WRONG: Storing Messages Before Streaming

```python
# DON'T DO THIS - You don't have the response yet
await db.add_message('user', prompt)
await db.add_message('assistant', ???)  # What goes here?

for token in response.response_gen:
    yield token
```

#### ✅ CORRECT: Store After Streaming Completes

```python
# DO THIS - Accumulate response, then store
full_response = ""
for token in response.response_gen:
    full_response += token
    yield token

# Now we have the complete response
await db.add_message('user', prompt)
await db.add_message('assistant', full_response)
```

---

See also:

- [Querying](./querying.md)
- [Configuration](./configuration.md)
- [Multi-Tenancy](./multi-tenancy.md)
