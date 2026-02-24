# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable


class ModelOutputParser(ABC):
    """Abstract base class for parsing model-specific output."""
    
    @abstractmethod
    def parse_output(self, json_data: dict) -> str:
        """Parse the output from a model response."""
        pass


class NovaOutputParser(ModelOutputParser):
    """Output parser for Amazon Nova models."""
    
    def parse_output(self, json_data: dict) -> str:
        contents = json_data.get('modelOutput', {}).get('output', {}).get('message', {}).get('content', [])
        text_parts = []
        for content in contents:
            if 'text' in content:
                text_parts.append(content['text'])
            # Skip reasoning content (redacted)
        return ''.join(text_parts)


class ClaudeOutputParser(ModelOutputParser):
    """Output parser for Anthropic Claude models."""
    
    def parse_output(self, json_data: dict) -> str:
        contents = json_data.get('modelOutput', {}).get('content', [])
        return ''.join([content.get('text', '') for content in contents])


class LlamaOutputParser(ModelOutputParser):
    """Output parser for Meta Llama models."""
    
    def parse_output(self, json_data: dict) -> str:
        return json_data['generation']


class ModelOutputParserFactory:
    """Factory for creating model-specific output parsers."""
    
    _parsers = {
        'amazon.nova': NovaOutputParser(),
        'anthropic.claude': ClaudeOutputParser(),
        'meta.llama': LlamaOutputParser(),
    }
    
    @classmethod
    def get_parser(cls, model_id: str) -> ModelOutputParser:
        """Get the appropriate parser for a model ID."""
        for key, parser in cls._parsers.items():
            if key in model_id:
                return parser
        raise ValueError(f'Unrecognized model_id: batch extraction for {model_id} is not supported')
    
    @classmethod
    def get_parse_function(cls, model_id: str) -> Callable[[dict], str]:
        """Get a parse function for backward compatibility."""
        parser = cls.get_parser(model_id)
        return parser.parse_output
    
    @classmethod
    def register_parser(cls, model_prefix: str, parser: ModelOutputParser) -> None:
        """Register a new parser for a model prefix (extension point)."""
        cls._parsers[model_prefix] = parser
