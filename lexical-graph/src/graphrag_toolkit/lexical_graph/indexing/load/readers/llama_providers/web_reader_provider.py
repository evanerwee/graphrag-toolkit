"""
Web page reader provider using LlamaIndex.

This module provides a provider that reads content from web pages
using LlamaIndex's SimpleWebPageReader.
"""

from typing import List, Callable, Optional, Dict, Any
from llama_index.core.schema import Document

from graphrag_toolkit.lexical_graph.indexing.load.readers.llama_index_reader_provider_base import (
    LlamaIndexReaderProviderBase,
)
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)


class WebReaderProvider(LlamaIndexReaderProviderBase):
    """
    Reader provider for web pages using LlamaIndex's SimpleWebPageReader.
    """

    def __init__(
        self,
        html_to_text: bool = True,
        metadata_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        **reader_kwargs
    ):
        """
        Initialize the WebReaderProvider.

        Args:
            html_to_text: Whether to convert HTML to plain text
            metadata_fn: Callable that returns metadata for a given URL
            reader_kwargs: Additional arguments passed to SimpleWebPageReader
        """
        try:
            from llama_index.readers.web import SimpleWebPageReader
        except ImportError as e:
            raise ImportError(
                "The 'llama-index[web]' optional dependency is required for WebReaderProvider.\n"
                "Install it with: pip install llama-index[web]"
            ) from e

        kwargs = {
            "html_to_text": html_to_text,
            "metadata_fn": metadata_fn or (lambda url: {"url": url}),
            **reader_kwargs,
        }

        super().__init__(reader_cls=SimpleWebPageReader, **kwargs)

    def self_test(self) -> bool:
        """
        Sanity check: Attempts to read a known simple web page.

        Returns:
            True if documents were loaded
        """
        test_url = "https://example.com"
        try:
            docs: List[Document] = self.read([test_url])
            assert isinstance(docs, list)
            return len(docs) > 0
        except Exception as e:
            logger.error(f"WebReaderProvider self_test failed: {e}")
            return False
