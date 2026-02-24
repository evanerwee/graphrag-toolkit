# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.anthropic.utils import messages_to_anthropic_messages
from llama_index.llms.bedrock_converse.utils import messages_to_converse_messages


class ModelRequestBuilder(ABC):
    """Abstract base class for building model-specific request bodies."""
    
    @abstractmethod
    def build_request(self, messages: List[ChatMessage], inference_parameters: dict) -> Dict[str, Any]:
        """Build a request body for the specific model."""
        pass


class NovaRequestBuilder(ModelRequestBuilder):
    """Request builder for Amazon Nova models."""
    
    def build_request(self, messages: List[ChatMessage], inference_parameters: dict) -> Dict[str, Any]:
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        request_body = {
            'messages': converse_messages,
            'inferenceConfig': {
                'maxTokens': inference_parameters['max_tokens'],
                'temperature': inference_parameters['temperature'],
            }
        }
        if system_prompt:
            request_body['system'] = [{'text': system_prompt}]
        
        # Extension: reasoning configuration
        if inference_parameters.get('reasoning_enabled', False):
            request_body['additionalModelRequestFields'] = {
                'reasoningConfig': {
                    'type': 'enabled',
                    'maxReasoningBudget': inference_parameters.get('reasoning_budget', 'low')
                }
            }
        
        return request_body


class ClaudeRequestBuilder(ModelRequestBuilder):
    """Request builder for Anthropic Claude models."""
    
    def build_request(self, messages: List[ChatMessage], inference_parameters: dict) -> Dict[str, Any]:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        if system_prompt:
            anthropic_messages = [{'role': 'system', 'content': system_prompt}, *anthropic_messages]
        return {
            'anthropic_version': inference_parameters.get('anthropic_version', 'bedrock-2023-05-31'),
            'messages': anthropic_messages,
            'max_tokens': inference_parameters['max_tokens'],
            'temperature': inference_parameters['temperature']
        }


class LlamaRequestBuilder(ModelRequestBuilder):
    """Request builder for Meta Llama models."""
    
    def build_request(self, messages: List[ChatMessage], inference_parameters: dict) -> Dict[str, Any]:
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        return {
            'messages': converse_messages,
            'parameters': {
                'max_new_tokens': inference_parameters['max_tokens'],
                'temperature': inference_parameters['temperature'],
            }
        }


class ModelRequestBuilderFactory:
    """Factory for creating model-specific request builders."""
    
    _builders: Dict[str, ModelRequestBuilder] = {
        'amazon.nova': NovaRequestBuilder(),
        'anthropic.claude': ClaudeRequestBuilder(),
        'meta.llama': LlamaRequestBuilder(),
    }
    
    @classmethod
    def get_builder(cls, model_id: str) -> ModelRequestBuilder:
        """Get the appropriate builder for a model ID."""
        for key, builder in cls._builders.items():
            if key in model_id:
                return builder
        raise ValueError(f'Unrecognized model_id: batch extraction for {model_id} is not supported')
    
    @classmethod
    def register_builder(cls, model_prefix: str, builder: ModelRequestBuilder) -> None:
        """Register a new builder for a model prefix (extension point)."""
        cls._builders[model_prefix] = builder
