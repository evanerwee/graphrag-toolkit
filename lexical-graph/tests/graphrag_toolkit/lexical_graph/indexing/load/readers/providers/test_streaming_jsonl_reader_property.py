# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Property-based tests for StreamingJSONLReaderProvider using Hypothesis framework.
These tests generate random inputs to discover edge cases and ensure robustness.
"""

import pytest
import json
import tempfile
import os
from hypothesis import given, strategies as st, assume, settings
from graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider import (
    StreamingJSONLReaderProvider,
    StreamingJSONLReaderConfig
)

# Skip these tests if hypothesis is not available
hypothesis = pytest.importorskip("hypothesis")


class TestStreamingJSONLReaderProviderProperties:
    """Property-based tests for StreamingJSONLReaderProvider."""

    @given(
        batch_size=st.integers(min_value=1, max_value=1000),
        strict_mode=st.booleans(),
        log_interval=st.integers(min_value=1, max_value=50000)
    )
    def test_config_initialization_properties(self, batch_size, strict_mode, log_interval):
        """Test that any valid configuration values work correctly."""
        config = StreamingJSONLReaderConfig(
            batch_size=batch_size,
            strict_mode=strict_mode,
            log_interval=log_interval
        )
        provider = StreamingJSONLReaderProvider(config)
        
        assert provider.batch_size == batch_size
        assert provider.strict_mode == strict_mode
        assert provider.log_interval == log_interval

    @given(
        json_objects=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                values=st.one_of(
                    st.text(min_size=0, max_size=100),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans()
                ),
                min_size=1,
                max_size=10
            ),
            min_size=1,
            max_size=100
        ),
        batch_size=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=5000)  # Limit examples for performance
    def test_batch_processing_properties(self, json_objects, batch_size):
        """Test that batch processing works correctly with any valid JSON data."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for obj in json_objects:
                f.write(json.dumps(obj) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(batch_size=batch_size)
            provider = StreamingJSONLReaderProvider(config)
            
            # Process in batches
            all_batches = list(provider.lazy_load_data(temp_file))
            
            # Verify batch properties
            total_docs = sum(len(batch) for batch in all_batches)
            assert total_docs == len(json_objects)
            
            # All batches except possibly the last should be full
            for i, batch in enumerate(all_batches[:-1]):
                assert len(batch) == batch_size
            
            # Last batch should have remaining documents
            if all_batches:
                expected_last_batch_size = len(json_objects) % batch_size
                if expected_last_batch_size == 0:
                    expected_last_batch_size = batch_size
                assert len(all_batches[-1]) == expected_last_batch_size
            
        finally:
            os.unlink(temp_file)

    @given(
        text_field=st.one_of(
            st.none(),
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))
        ),
        json_objects=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                values=st.text(min_size=1, max_size=100),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=15, deadline=5000)
    def test_text_field_extraction_properties(self, text_field, json_objects):
        """Test text field extraction with various field names and JSON structures."""
        # Ensure at least some objects have the text field if specified
        if text_field is not None:
            for i, obj in enumerate(json_objects[:len(json_objects)//2]):
                obj[text_field] = f"Content for object {i}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for obj in json_objects:
                f.write(json.dumps(obj) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(text_field=text_field, strict_mode=False)
            provider = StreamingJSONLReaderProvider(config)
            
            docs = provider.load_data(temp_file)
            
            # Verify text extraction
            for doc in docs:
                if text_field is None:
                    # Should be full JSON
                    parsed = json.loads(doc.text)
                    assert isinstance(parsed, dict)
                else:
                    # Should be the text field value or skipped if missing
                    assert isinstance(doc.text, str)
                    assert len(doc.text) > 0
            
        finally:
            os.unlink(temp_file)

    @given(
        valid_lines=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'),
                values=st.text(min_size=1, max_size=50),
                min_size=1,
                max_size=3
            ),
            min_size=0,
            max_size=20
        ),
        invalid_lines=st.lists(
            st.one_of(
                st.text(min_size=1, max_size=50).filter(lambda x: not x.strip().startswith('{')),
                st.just('{"incomplete": json'),
                st.just('not json at all'),
                st.just(''),
                st.just('   ')
            ),
            min_size=0,
            max_size=10
        ),
        strict_mode=st.booleans()
    )
    @settings(max_examples=10, deadline=5000)
    def test_error_handling_properties(self, valid_lines, invalid_lines, strict_mode):
        """Test error handling with mix of valid and invalid JSON lines."""
        # Interleave valid and invalid lines
        all_lines = []
        for i in range(max(len(valid_lines), len(invalid_lines))):
            if i < len(valid_lines):
                all_lines.append(json.dumps(valid_lines[i]))
            if i < len(invalid_lines):
                all_lines.append(invalid_lines[i])
        
        if not all_lines:
            all_lines = ['{"test": "data"}']  # Ensure at least one line
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for line in all_lines:
                f.write(line + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(strict_mode=strict_mode)
            provider = StreamingJSONLReaderProvider(config)
            
            if strict_mode and invalid_lines:
                # Should raise exception on first invalid line
                with pytest.raises((ValueError, RuntimeError)):
                    provider.load_data(temp_file)
            else:
                # Should process valid lines and skip invalid ones
                docs = provider.load_data(temp_file)
                # Should have at least as many docs as valid lines (might have more if invalid lines are valid JSON)
                assert len(docs) >= len(valid_lines)
                
        finally:
            os.unlink(temp_file)

    @given(
        metadata_keys=st.lists(
            st.text(min_size=1, max_size=15, alphabet='abcdefghijklmnopqrstuvwxyz_'),
            min_size=1,
            max_size=10,
            unique=True
        ),
        metadata_values=st.lists(
            st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.lists(st.text(max_size=20), max_size=5),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'),
                    values=st.text(max_size=20),
                    max_size=3
                )
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_metadata_handling_properties(self, metadata_keys, metadata_values):
        """Test metadata handling with various JSON structures."""
        assume(len(metadata_keys) == len(metadata_values))
        
        # Create JSON objects with the generated metadata
        json_objects = []
        for i in range(min(5, len(metadata_keys))):  # Limit to 5 objects for performance
            obj = {}
            for j, (key, value) in enumerate(zip(metadata_keys, metadata_values)):
                if j <= i:  # Gradually add more fields
                    obj[key] = value
            json_objects.append(obj)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for obj in json_objects:
                f.write(json.dumps(obj) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig()
            provider = StreamingJSONLReaderProvider(config)
            
            docs = provider.load_data(temp_file)
            
            # Verify metadata is properly extracted
            assert len(docs) == len(json_objects)
            
            for doc in docs:
                # Should have basic metadata
                assert 'source_path' in doc.metadata
                assert 'line_number' in doc.metadata
                assert 'reader_type' in doc.metadata
                
                # Should have JSON fields as metadata
                for key in metadata_keys:
                    json_key = f'json_{key}'
                    # Metadata might not be present if the specific object didn't have this key
                    if json_key in doc.metadata:
                        assert isinstance(doc.metadata[json_key], str)
                        
        finally:
            os.unlink(temp_file)

    @given(
        line_count=st.integers(min_value=1, max_value=1000),
        batch_size=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=10, deadline=5000)
    def test_batch_count_properties(self, line_count, batch_size):
        """Test that batch count calculations are always correct."""
        # Create file with exact number of lines
        json_objects = [{"id": i, "content": f"Document {i}"} for i in range(line_count)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for obj in json_objects:
                f.write(json.dumps(obj) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(batch_size=batch_size)
            provider = StreamingJSONLReaderProvider(config)
            
            batches = list(provider.lazy_load_data(temp_file))
            
            # Calculate expected batch count
            expected_batch_count = (line_count + batch_size - 1) // batch_size  # Ceiling division
            assert len(batches) == expected_batch_count
            
            # Verify total document count
            total_docs = sum(len(batch) for batch in batches)
            assert total_docs == line_count
            
            # Verify batch sizes
            for i, batch in enumerate(batches):
                if i < len(batches) - 1:
                    # All batches except last should be full
                    assert len(batch) == batch_size
                else:
                    # Last batch should have remaining documents
                    expected_last_size = line_count % batch_size
                    if expected_last_size == 0:
                        expected_last_size = batch_size
                    assert len(batch) == expected_last_size
                    
        finally:
            os.unlink(temp_file)

    @given(
        empty_line_positions=st.lists(
            st.integers(min_value=0, max_value=19),
            max_size=10,
            unique=True
        ),
        total_lines=st.integers(min_value=10, max_value=20)
    )
    @settings(max_examples=5, deadline=3000)
    def test_empty_line_handling_properties(self, empty_line_positions, total_lines):
        """Test handling of empty lines at various positions."""
        lines = []
        for i in range(total_lines):
            if i in empty_line_positions:
                lines.append('')  # Empty line
            else:
                lines.append(json.dumps({"id": i, "content": f"Content {i}"}))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for line in lines:
                f.write(line + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig()
            provider = StreamingJSONLReaderProvider(config)
            
            docs = provider.load_data(temp_file)
            
            # Should have documents only for non-empty lines
            expected_doc_count = total_lines - len(empty_line_positions)
            assert len(docs) == expected_doc_count
            
            # All documents should have valid content
            for doc in docs:
                assert doc.text is not None
                assert len(doc.text.strip()) > 0
                
        finally:
            os.unlink(temp_file)