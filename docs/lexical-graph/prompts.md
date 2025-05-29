
## Using Custom Prompt Providers

The GraphRAG Toolkit supports pluggable prompt providers to allow dynamic loading of prompt templates from various sources. There are four built-in providers:

### 1. StaticPromptProvider

Use this when your system and user prompts are defined as constants in your codebase.

```python
from graphrag_toolkit.lexical_graph.prompts.static_prompt_provider import StaticPromptProvider

prompt_provider = StaticPromptProvider()
```

This provider uses the predefined constants `ANSWER_QUESTION_SYSTEM_PROMPT` and `ANSWER_QUESTION_USER_PROMPT`.

---

### 2. FilePromptProvider

Use this when your prompts are stored locally on disk.

```python
from graphrag_toolkit.lexical_graph.prompts.file_prompt_provider import FilePromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import FilePromptProviderConfig

prompt_provider = FilePromptProvider(
    FilePromptProviderConfig(base_path="./prompts"),
    system_prompt_file="system.txt",
    user_prompt_file="user.txt"
)
```

The prompt files are read from a directory (`base_path`), and you can override the file names if needed.

---

### 3. S3PromptProvider

Use this when your prompts are stored in an Amazon S3 bucket.

```python
from graphrag_toolkit.lexical_graph.prompts.s3_prompt_provider import S3PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import S3PromptProviderConfig

prompt_provider = S3PromptProvider(
    S3PromptProviderConfig(
        bucket="ccms-prompts",
        prefix="prompts",
        aws_region="us-east-1",        # optional if set via env
        aws_profile="my-profile",      # optional if using default profile
        system_prompt_file="my_system.txt",  # optional override
        user_prompt_file="my_user.txt"       # optional override
    )
)
```

Prompts are loaded using `boto3` and AWS credentials. Ensure your environment or `~/.aws/config` is configured for SSO, roles, or keys.

---

### 4. BedrockPromptProvider

Use this when your prompts are stored and versioned using Amazon Bedrock prompt ARNs.

```python
from graphrag_toolkit.lexical_graph.prompts.bedrock_prompt_provider import BedrockPromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import BedrockPromptProviderConfig

prompt_provider = BedrockPromptProvider(
    config=BedrockPromptProviderConfig(
        system_prompt_arn="arn:aws:bedrock:us-east-1:123456789012:prompt/my-system",
        user_prompt_arn="arn:aws:bedrock:us-east-1:123456789012:prompt/my-user",
        system_prompt_version="DRAFT",
        user_prompt_version="DRAFT"
    )
)
```

This provider resolves prompt ARNs dynamically using STS and can fall back to environment variables if needed.

---

Would you like this formatted into a Markdown file or added directly into your Sphinx docs structure (`index.md` or a separate `prompts.md`)?
