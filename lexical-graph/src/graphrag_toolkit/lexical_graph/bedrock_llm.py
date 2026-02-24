# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Direct Bedrock LLM wrapper that bypasses LlamaIndex's model validation.

This module provides a custom LLM implementation that uses boto3's bedrock-runtime
client directly, allowing support for any Bedrock model without hardcoded validation.
"""

import json
import logging
from typing import Any, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.callbacks import CallbackManager

logger = logging.getLogger(__name__)


class DirectBedrockLLM(LLM):
    """
    Direct Bedrock LLM that uses boto3 bedrock-runtime client.
    
    This implementation bypasses LlamaIndex's model name validation,
    allowing any Bedrock model to be used including Nova 2 series.
    
    The client is created lazily and recreated after unpickling to handle
    multiprocessing scenarios where boto3 sessions cannot be pickled.
    
    Attributes:
        model: The Bedrock model ID (e.g., 'amazon.nova-2-lite-v1:0')
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
    """
    
    model: str = Field(description="Bedrock model ID")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    
    _client: Any = PrivateAttr(default=None)
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DirectBedrockLLM."""
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            callback_manager=callback_manager,
            **kwargs,
        )
        
        self._client = None
        
        logger.info(f"[DirectBedrockLLM] Initialized with model: {model}")
    
    def __getstate__(self):
        """Custom pickle support - exclude the client."""
        state = super().__getstate__()
        # Remove the client from the pickled state
        if '__pydantic_private__' in state and '_client' in state['__pydantic_private__']:
            state['__pydantic_private__']['_client'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickle support - client will be recreated on first use."""
        super().__setstate__(state)
        self._client = None
    
    @property
    def client(self):
        """
        Lazy initialization of bedrock-runtime client.
        
        Gets the client from GraphRAGConfig's session to ensure proper
        credential management across different environments (local, EKS, SSO, etc.).
        """
        if self._client is None:
            # Import here to avoid circular dependency
            from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
            
            # Use GraphRAGConfig's session which handles all credential scenarios
            session = GraphRAGConfig.session
            self._client = session.client('bedrock-runtime')
            logger.debug(f"[DirectBedrockLLM] Created bedrock-runtime client from GraphRAGConfig.session")
        
        return self._client
    
    @classmethod
    def class_name(cls) -> str:
        """Return class name."""
        return "DirectBedrockLLM"
    
    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        # Provide reasonable defaults for context window
        # Nova models typically support 300K tokens
        return LLMMetadata(
            context_window=300000,
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )
    
    def _prepare_messages(self, messages: Sequence[ChatMessage]) -> list:
        """Convert LlamaIndex ChatMessage to Bedrock Converse format."""
        bedrock_messages = []
        
        for msg in messages:
            role = msg.role
            # Map LlamaIndex roles to Bedrock roles
            if role == "system":
                # System messages handled separately in Converse API
                continue
            elif role == "assistant":
                bedrock_role = "assistant"
            else:
                bedrock_role = "user"
            
            bedrock_messages.append({
                "role": bedrock_role,
                "content": [{"text": msg.content}]
            })
        
        return bedrock_messages
    
    def _extract_system_message(self, messages: Sequence[ChatMessage]) -> Optional[str]:
        """Extract system message from messages."""
        for msg in messages:
            if msg.role == "system":
                return msg.content
        return None
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Synchronous chat completion."""
        bedrock_messages = self._prepare_messages(messages)
        system_message = self._extract_system_message(messages)
        
        # Prepare Converse API request
        request_params = {
            "modelId": self.model,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            }
        }
        
        if system_message:
            request_params["system"] = [{"text": system_message}]
        
        logger.debug(f"[DirectBedrockLLM] Calling Bedrock Converse API with model: {self.model}")
        
        try:
            response = self.client.converse(**request_params)
            
            # Extract response text
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            response_text = ""
            for block in content_blocks:
                if "text" in block:
                    response_text += block["text"]
            
            # Extract usage metadata
            usage = response.get("usage", {})
            
            return ChatResponse(
                message=ChatMessage(role="assistant", content=response_text),
                raw=response,
                additional_kwargs={
                    "usage": usage,
                    "stop_reason": response.get("stopReason"),
                }
            )
        
        except Exception as e:
            logger.error(f"[DirectBedrockLLM] Error calling Bedrock: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Synchronous completion (converts to chat format)."""
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = self.chat(messages, **kwargs)
        
        return CompletionResponse(
            text=chat_response.message.content,
            raw=chat_response.raw,
            additional_kwargs=chat_response.additional_kwargs,
        )
    
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat (not implemented - falls back to non-streaming)."""
        logger.warning("[DirectBedrockLLM] Streaming not implemented, using non-streaming chat")
        response = self.chat(messages, **kwargs)
        yield response
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Streaming completion (not implemented - falls back to non-streaming)."""
        logger.warning("[DirectBedrockLLM] Streaming not implemented, using non-streaming complete")
        response = self.complete(prompt, **kwargs)
        yield response
    
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat (falls back to sync)."""
        return self.chat(messages, **kwargs)
    
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion (falls back to sync)."""
        return self.complete(prompt, **kwargs)
    
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Async streaming chat (falls back to sync)."""
        response = await self.achat(messages, **kwargs)
        yield response
    
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        """Async streaming completion (falls back to sync)."""
        response = await self.acomplete(prompt, **kwargs)
        yield response
