
# Using Custom Reader Providers

The GraphRAG Toolkit supports pluggable **reader providers** to allow structured ingestion of content from multiple source types such as local directories, S3 buckets, web pages, PDFs, GitHub repos, and more.

Each provider implements the `ReaderProvider` interface and can be dynamically loaded via the control plane or instantiated directly in code.

---

## Provider Categories

Reader providers are currently grouped into one category:

* **Llama Reader Providers** – wrappers around LlamaIndex readers (e.g. PDF, web, docx, directory, etc.)

---

## Llama Reader Providers

> All of these live under  
> `graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers`

---

### 1. DirectoryReaderProvider

Use this to read a local directory containing mixed file types (`.pdf`, `.docx`, `.html`, etc.).

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.directory_reader_provider import DirectoryReaderProvider

# Point at your local folder...
provider = DirectoryReaderProvider(data_dir="data/mixed-files/")
docs = provider.read()

# Then feed into your graph index
graph_index.extract_and_build(docs, show_progress=True)
````

---

### 2. WebReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.web_reader_provider import WebReaderProvider

urls = [
    "https://aws.amazon.com/neptune/",
    "https://aws.amazon.com/neptune-analytics/"
]

provider = WebReaderProvider()
docs = provider.read(urls)
graph_index.extract_and_build(docs, show_progress=True)
```

---

### 3. PDFReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.pdf_reader_provider import PDFReaderProvider

provider = PDFReaderProvider()
docs = provider.read("docs/sample.pdf")
```

---

### 4. YouTubeReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.youtube_reader_provider import YouTubeReaderProvider

provider = YouTubeReaderProvider()
docs = provider.read("https://www.youtube.com/watch?v=YmR2_zlQO5w")
```

---

### 5. S3DirectoryReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import S3DirectoryReaderConfig
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.s3_directory_reader_provider import S3DirectoryReaderProvider

config = S3DirectoryReaderConfig(
    bucket="my-bucket",
    prefix="documents/",
    region="us-east-1",
    profile=None       # or your AWS profile name
)

provider = S3DirectoryReaderProvider(config=config)
docs = provider.read()
```

---

### 6. GitHubReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.github_repo_provider import GitHubReaderProvider
import os

repo_url    = "https://github.com/aws/graphrag-toolkit"
branch      = "main"
github_token = os.getenv("GITHUB_TOKEN")

provider = GitHubReaderProvider(repo_url=repo_url, branch=branch, token=github_token)
docs = provider.read()
```

---

### 7. DocxReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.docx_reader_provider import DocxReaderProvider

provider = DocxReaderProvider()
docs = provider.read("docs/story.docx")
```

---

### 8. PPTXReaderProvider

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.pptx_reader_provider import PPTXReaderProvider

provider = PPTXReaderProvider()
docs = provider.read("presentations/sample.pptx")
```

---

## Configuration-Driven Instantiation

You can also wire readers up via your control-plane config:

```python
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import ReaderProviderConfig
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_factory import get_reader_provider

config = ReaderProviderConfig(type="web", params={})
provider = get_reader_provider(config)
docs = provider.read(["https://example.com"])
```

---

## Available Reader Provider Types

| Provider Type  | Class                       |
| -------------- | --------------------------- |
| `directory`    | `DirectoryReaderProvider`   |
| `s3_directory` | `S3DirectoryReaderProvider` |
| `pdf`          | `PDFReaderProvider`         |
| `web`          | `WebReaderProvider`         |
| `youtube`      | `YouTubeReaderProvider`     |
| `github`       | `GitHubReaderProvider`      |
| `docx`         | `DocxReaderProvider`        |
| `pptx`         | `PPTXReaderProvider`        |

---

## Full Example: Web → GraphRAG Indexing

```python
import os
from graphrag_toolkit.lexical_graph import LexicalGraphIndex, set_logging_config
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory, VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph.falkordb import FalkorDBGraphStoreFactory
from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_providers.web_reader_provider import WebReaderProvider

# register your graph store
GraphStoreFactory.register(FalkorDBGraphStoreFactory)

graph_store  = GraphStoreFactory.for_graph_store(os.environ["GRAPH_STORE"])
vector_store = VectorStoreFactory.for_vector_store(os.environ["VECTOR_STORE"])
graph_index  = LexicalGraphIndex(graph_store, vector_store)

# read some URLs
provider = WebReaderProvider()
docs     = provider.read([
    "https://aws.amazon.com/neptune/",
    "https://aws.amazon.com/neptune-analytics/"
])

# extract and build your RAG index
graph_index.extract_and_build(docs, show_progress=True)
print("Indexing complete!")
```

```
```
