# AWS Prompt Provider Architecture

## Overview

This document outlines the design rationale and implementation of a flexible, pluggable prompt retrieval system within the GraphRAG Toolkit. The objective is to decouple prompt configuration from the static `GraphRAGConfig` and allow runtime injection of prompt templates sourced from various backends.

This design supports dynamic selection of prompt providers depending on the environment (development, staging, production) and use case.

## Prompt Provider Design

### PromptProvider (Abstract Base Class)

The `PromptProvider` class defines the interface for retrieving prompts:

```python
class PromptProvider(ABC):
    def get_system_prompt(self) -> Optional[str]: ...
    def get_user_prompt(self) -> Optional[str]: ...
```

### Implementations

* **BedrockPromptProvider**

  * Retrieves prompt templates from AWS Bedrock's prompt registry.
  * Uses `GraphRAGConfig.session` and region.
  * Environment variables `SYSTEM_PROMPT_ARN` and `USER_PROMPT_ARN` are used to identify prompt resources.

* **S3PromptProvider**

  * Retrieves prompts from versioned text files in an S3 bucket.
  * Useful for staging and managed prompt workflows via Git.
  * Configured using bucket and prefix.

* **FileSystemPromptProvider**

  * Loads prompts from the local filesystem.
  * Useful for development and testing.

## ARN Construction

Bedrock ARNs are resolved dynamically using STS:

```python
identity = sts.get_caller_identity()
partition = identity["Arn"].split(":")[1]  # e.g., aws, aws-cn, aws-us-gov
account_id = identity["Account"]
```

This ensures compatibility across partitions and AWS environments (e.g., GovCloud).

## SOLID Principles Applied

* **Single Responsibility**: Each provider class is responsible for retrieving prompts from one source.
* **Open/Closed**: New providers can be added without modifying existing logic.
* **Liskov Substitution**: All providers adhere to the same abstract base interface.
* **Interface Segregation**: Consumers use only the methods they need (`get_system_prompt` / `get_user_prompt`).
* **Dependency Inversion**: Application logic depends on the abstract `PromptProvider` rather than concrete classes.

## Security & Permissions

### Bedrock

* Requires `bedrock:GetPrompt` permissions.
* Ensure access to prompt resource ARNs.

### S3

* Requires `s3:GetObject` permissions on the configured bucket and key prefix.
* Least-privilege IAM policies are recommended.

## Usage Summary

Prompt providers should be instantiated and injected during construction of the query engine or retrieval pipeline. This promotes flexibility, security, and clean separation of concerns.

Future extensions may include:

* Parameterized prompts with placeholder substitution
* Prompt versioning
* Prompt registry via DynamoDB or API Gateway
