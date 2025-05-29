from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import BedrockPromptProviderConfig
from graphrag_toolkit.lexical_graph.prompts.bedrock_prompt_provider import BedrockPromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider import PromptProvider

import boto3
import os

def build_bedrock_prompt_provider() -> PromptProvider:
    session = boto3.Session()
    region = os.getenv("AWS_REGION", "us-east-1")

    config = BedrockPromptProviderConfig.from_session(session, region)

    system_arn = os.environ["SYSTEM_PROMPT_ARN"]
    user_arn = os.environ["USER_PROMPT_ARN"]

    return BedrockPromptProvider(
        config=config,
        system_prompt_arn=system_arn,
        user_prompt_arn=user_arn,
    )
