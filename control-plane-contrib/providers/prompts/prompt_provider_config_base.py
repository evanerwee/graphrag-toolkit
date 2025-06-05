# prompt_provider_config_base.py
from pydantic import BaseModel
from typing import Optional

class PromptFormatMixin(BaseModel):
    """
    An optional mixin for specifying prompt format.
    """
    format: Optional[str] = "text"  # Allowed: "text", "json"

class FilePromptProviderConfig(PromptFormatMixin,BaseModel):
    """
    Configuration model for file-based prompt providers.

    This class defines the required fields for specifying system and user prompt file names.
    """
    system_prompt_file: str
    user_prompt_file: str

class S3PromptProviderConfig(PromptFormatMixin,BaseModel):
    """
    Configuration model for S3-based prompt providers.

    This class defines the required fields for specifying the S3 bucket, key, and optional region for prompt storage.
    """
    bucket: str
    key: str
    region: Optional[str] = None

class BedrockPromptProviderConfig(PromptFormatMixin,BaseModel):
    """
    Configuration model for Bedrock-based prompt providers.

    This class defines the required field for specifying the Bedrock prompt ARN.
    """
    prompt_arn: str

class DynamoDBPromptProviderConfig(PromptFormatMixin,BaseModel):
    """
    Configuration model for DynamoDB-based prompt providers.

    Attributes:
        table_name: Name of the DynamoDB table.
        tenant_id: Partition key identifying the tenant.
        system_prompt_key: Sort key (or attribute key) for the system prompt.
        user_prompt_key: Sort key (or attribute key) for the user prompt.
        region: AWS region where the DynamoDB table resides.
    """
    table_name: str
    tenant_id: str
    system_prompt_key: str
    user_prompt_key: str
    region: Optional[str] = None
    profile_name: Optional[str] = None



