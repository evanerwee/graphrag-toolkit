
---

## GraphRAG Prompts V2 (Control Plane) – Provider-Based Architecture

Prompts V2 introduces an extensible provider model that supports multiple prompt formats (`text`, `json`) and backends (local, cloud, managed services). This architecture aligns with **SOLID principles** and supports **control-plane driven prompt selection** per tenant or use-case.

---

### Supported Providers

| Provider   | Description                                      | Format Support |
| ---------- | ------------------------------------------------ | -------------- |
| `Static`   | Built-in fallback using hardcoded prompt strings | `text` only    |
| `File`     | Load prompts from local files                    | `text`, `json` |
| `S3`       | Load prompts from Amazon S3                      | `text`, `json` |
| `Bedrock`  | Load prompt templates from AWS Bedrock           | `text`, `json` |
| `DynamoDB` | Load tenant prompts from structured DDB tables   | `text`, `json` |

---

### PromptProvider Interface (V2)

Each provider implements:

```python
class PromptProvider(ABC):
    def get_system_prompt(self) -> str | dict:
        ...
    def get_user_prompt(self) -> str | dict:
        ...
```

If `format="json"` is configured, the result is a `dict` (structured prompt template). Otherwise, a plain string is returned.

---

### Configuration via PromptProviderConfig

Each provider has its own config class in `prompt_provider_config.py`. Example:

```python
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import FilePromptProviderConfig

config = FilePromptProviderConfig(
    base_path="./prompts",
    system_prompt_file="system.json",
    user_prompt_file="user.json",
    format="json"
)
```

You may also use `BedrockPromptProviderConfig`, `S3PromptProviderConfig`, or `DynamoDBPromptProviderConfig`.

---

### PromptProviderFactory (Dynamic Loading)

```python
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_factory import PromptProviderFactory

prompt_provider = PromptProviderFactory.get_provider()
prompt_provider.get_system_prompt()
```

Set the environment variable `PROMPT_PROVIDER=[file|s3|bedrock|dynamodb|static]`.

---

### Format Inference and Fallback

If `format` is **not explicitly specified**:

* `FilePromptProvider` and `S3PromptProvider` **auto-detect** from file extension (`.json` → JSON mode)
* All other providers default to `"text"`

---

### BedrockPromptProvider Enhancements

* Supports both `text` and `json` blocks in the Bedrock `templateConfiguration`
* Resolves short names to ARNs using partition-aware logic (`arn:aws`, `arn:aws-cn`, etc.)

---

### DynamoDBPromptProvider Schema (V2)

| PK (tenant\_id) | SK (prompt\_type#id) | format | content        |
| --------------- | -------------------- | ------ | -------------- |
| `tenant-001`    | `system#default`     | `json` | JSON blob/text |
| `tenant-001`    | `user#v1`            | `text` | string prompt  |

Supports:

* Per-tenant isolation
* Optional CLI loader
* Versioning via sort key

---

### CLI Prompt Loader (Optional Tool)

Load prompts via:

```bash
bash scripts/load_prompts_dynamodb.sh --profile myprofile --table PromptTable \
  --tenant acme --type system --file ./prompts/acme_system.json --format json
```

---

### PromptProviderRegistry

To support dynamic lookup across tenants:

```python
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_registry import PromptProviderRegistry

PromptProviderRegistry.register("tenant-acme", provider)
provider = PromptProviderRegistry.get("tenant-acme")
```

---

### Usage in QueryEngine

```python
query_engine = LexicalGraphQueryEngine.for_semantic_guided_search(
    graph_store,
    vector_store,
    retrievers=[...],
    prompt_provider=prompt_provider  # Inject provider here
)
```

---

### Future V3 Concepts (Not in V2)

* Prompt templating with dynamic variable injection
* Prompt scoring/evaluation
* Prompt version pinning & rollback
* Multi-language prompt selection (i18n)
* Prompt analytics and A/B testing

