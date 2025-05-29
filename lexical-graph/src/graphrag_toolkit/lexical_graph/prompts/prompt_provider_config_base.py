# prompt_provider_config_base.py
from pydantic import BaseModel
from typing import Optional

class FilePromptProviderConfig(BaseModel):
    system_prompt_file: str
    user_prompt_file: str

class S3PromptProviderConfig(BaseModel):
    bucket: str
    key: str
    region: Optional[str] = None

class BedrockPromptProviderConfig(BaseModel):
    prompt_arn: str
