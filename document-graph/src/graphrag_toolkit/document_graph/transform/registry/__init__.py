# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry module for transformer providers in document graph operations.

This package provides registry and factory components for managing transformer providers
in the document graph processing system. It includes classes for registering, discovering,
and instantiating transformer provider implementations.
"""

from graphrag_toolkit.document_graph.transform.registry.transformer_provider_registry import TransformerProviderRegistry
from graphrag_toolkit.document_graph.transform.registry.transformer_provider_factory import TransformerProviderFactory

# Create a singleton instance for backward compatibility
transformer_provider_registry = TransformerProviderRegistry()

__all__ = [
    "TransformerProviderRegistry",
    "TransformerProviderFactory",
    "transformer_provider_registry",
]