# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Streaming JSONL Reader Provider for memory-efficient processing of large JSONL files.
Provides batch-based streaming to handle files that exceed available memory.
"""

import json
import os
from typing import List, Optional, Iterator, Dict, Any, Callable
from llama_index.core.schema import Document
from ..base_reader_provider import BaseReaderProvider
from ..reader_provider_config_base import ReaderProviderConfig
from ..s3_file_mixin import S3FileMixin
from graphrag_toolkit.lexical_graph.logging import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreamingJSONLReaderConfig(ReaderProviderConfig):
    """Configuration for StreamingJSONLReaderProvider."""
    batch_size: int = 1000
    text_field: Optional[str] = None  # None = use full JSON as text
    strict_mode: bool = False
    log_interval: int = 10000
    metadata_fn: Optional[Callable[[str], Dict[str, Any]]] = None


class StreamingJSONLReaderProvider(BaseReaderProvider, S3FileMixin):
    """
    Memory-efficient streaming JSONL reader that processes files in batches.
    
    Features:
    - Constant memory usage regardless of file size
    - S3 support via S3FileMixin
    - Configurable batch processing
    - Error handling modes (strict vs lenient)
    - Progress logging for large files
    - Custom metadata extraction
    """

    def __init__(self, config: StreamingJSONLReaderConfig):
        """Initialize with StreamingJSONLReaderConfig."""
        super().__init__(config)
        self.batch_size = config.batch_size
        self.text_field = config.text_field
        self.strict_mode = config.strict_mode
        self.log_interval = config.log_interval
        self.metadata_fn = config.metadata_fn
        
        logger.debug(f"Initialized StreamingJSONLReaderProvider with batch_size={self.batch_size}, "
                    f"text_field={self.text_field}, strict_mode={self.strict_mode}")

    def load_data(self, input_source: str) -> List[Document]:
        """
        Load all documents from JSONL file into memory.
        For large files, consider using lazy_load_data instead.
        """
        if not input_source:
            logger.error("No input source provided to StreamingJSONLReaderProvider")
            raise ValueError("input_source cannot be None or empty")
        
        logger.info(f"Loading JSONL data from: {input_source}")
        
        all_documents = []
        for batch in self.lazy_load_data(input_source):
            all_documents.extend(batch)
        
        logger.info(f"Successfully loaded {len(all_documents)} document(s) from JSONL")
        return all_documents

    def lazy_load_data(self, input_source: str) -> Iterator[List[Document]]:
        """
        Lazily load documents in batches for memory-efficient processing.
        
        Args:
            input_source: Path to JSONL file (local or S3)
            
        Yields:
            List[Document]: Batches of documents
        """
        if not input_source:
            logger.error("No input source provided to StreamingJSONLReaderProvider")
            raise ValueError("input_source cannot be None or empty")
        
        logger.info(f"Streaming JSONL data from: {input_source}")
        processed_paths, temp_files, original_paths = self._process_file_paths(input_source)
        
        try:
            file_path = processed_paths[0]
            original_path = original_paths[0]
            
            batch: List[Document] = []
            line_count = 0
            document_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    line_count += 1
                    
                    # Log progress periodically
                    if line_count % self.log_interval == 0:
                        logger.info(f"Processed {line_count} lines, created {document_count} documents")
                    
                    doc = self._process_line(line, line_number, original_path)
                    
                    if doc is not None:
                        batch.append(doc)
                        document_count += 1
                        
                        # Yield batch when full
                        if len(batch) >= self.batch_size:
                            logger.debug(f"Yielding batch of {len(batch)} documents")
                            yield batch
                            batch = []
            
            # Yield final partial batch
            if batch:
                logger.debug(f"Yielding final batch of {len(batch)} documents")
                yield batch
            
            logger.info(f"Completed streaming: processed {line_count} lines, "
                       f"created {document_count} documents")
                       
        except Exception as e:
            logger.error(f"Failed to stream JSONL from {input_source}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to stream JSONL: {e}") from e
        finally:
            self._cleanup_temp_files(temp_files)

    def _process_line(self, line: str, line_number: int, source_path: str) -> Optional[Document]:
        """
        Parse a single JSONL line into a Document.
        
        Args:
            line: Raw line from JSONL file
            line_number: Line number for error reporting
            source_path: Original source path for metadata
            
        Returns:
            Document or None if line should be skipped
        """
        stripped_line = line.strip()
        if not stripped_line:
            return None
        
        try:
            json_obj = json.loads(stripped_line)
        except json.JSONDecodeError as e:
            if self.strict_mode:
                logger.error(f"Malformed JSON at line {line_number}: {e}")
                raise ValueError(f"Malformed JSON at line {line_number}: {e.msg}") from e
            else:
                logger.warning(f"Skipping line {line_number}: JSONDecodeError - {e.msg}")
                return None
        
        # Extract text based on configuration
        if self.text_field is None:
            # Use entire JSON object as text
            text = json.dumps(json_obj)
        elif self.text_field in json_obj:
            text = str(json_obj[self.text_field])
        else:
            if self.strict_mode:
                error_msg = f"Missing text_field '{self.text_field}' at line {line_number}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                logger.warning(f"Skipping line {line_number}: missing text_field '{self.text_field}'")
                return None
        
        # Validate text is not empty
        if not text or not text.strip():
            if self.strict_mode:
                error_msg = f"Empty text content at line {line_number}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                logger.warning(f"Skipping line {line_number}: empty text content")
                return None
        
        # Build metadata
        metadata = self._build_metadata(source_path, line_number, json_obj)
        
        return Document(text=text, metadata=metadata)

    def _build_metadata(self, source_path: str, line_number: int, json_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build metadata for a document.
        
        Args:
            source_path: Original source path
            line_number: Line number in file
            json_obj: Parsed JSON object
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'source': self._get_file_source_type(source_path),
            'source_path': source_path,
            'line_number': line_number,
            'reader_type': 'streaming_jsonl'
        }
        
        # Add custom metadata if function provided
        if self.metadata_fn:
            try:
                additional_metadata = self.metadata_fn(source_path)
                if additional_metadata:
                    metadata.update(additional_metadata)
            except Exception as e:
                logger.warning(f"Custom metadata function failed for {source_path}: {e}")
        
        # Add JSON fields as metadata (excluding text field to avoid duplication)
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if key != self.text_field:  # Don't duplicate text content in metadata
                    # Convert complex objects to strings for metadata
                    if isinstance(value, (dict, list)):
                        metadata[f'json_{key}'] = json.dumps(value)
                    else:
                        metadata[f'json_{key}'] = str(value)
        
        return metadata