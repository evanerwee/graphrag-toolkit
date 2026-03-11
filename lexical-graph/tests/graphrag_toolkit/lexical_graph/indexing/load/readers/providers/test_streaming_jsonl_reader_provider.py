# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider import (
    StreamingJSONLReaderProvider,
    StreamingJSONLReaderConfig
)


class TestStreamingJSONLReaderProvider:
    """Test suite for StreamingJSONLReaderProvider."""

    def test_initialization(self):
        """Test provider initialization with default config."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        assert provider.batch_size == 1000
        assert provider.text_field is None
        assert provider.strict_mode is False
        assert provider.log_interval == 10000
        assert provider.metadata_fn is None

    def test_initialization_with_custom_config(self):
        """Test provider initialization with custom configuration."""
        def custom_metadata(path):
            return {"custom": "metadata"}
        
        config = StreamingJSONLReaderConfig(
            batch_size=500,
            text_field="content",
            strict_mode=True,
            log_interval=5000,
            metadata_fn=custom_metadata
        )
        provider = StreamingJSONLReaderProvider(config)
        
        assert provider.batch_size == 500
        assert provider.text_field == "content"
        assert provider.strict_mode is True
        assert provider.log_interval == 5000
        assert provider.metadata_fn == custom_metadata

    def test_load_data_empty_input(self):
        """Test that empty input raises ValueError."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        with pytest.raises(ValueError, match="input_source cannot be None or empty"):
            provider.load_data("")
        
        with pytest.raises(ValueError, match="input_source cannot be None or empty"):
            provider.load_data(None)

    def test_lazy_load_data_empty_input(self):
        """Test that empty input raises ValueError for lazy loading."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        with pytest.raises(ValueError, match="input_source cannot be None or empty"):
            list(provider.lazy_load_data(""))
        
        with pytest.raises(ValueError, match="input_source cannot be None or empty"):
            list(provider.lazy_load_data(None))

    def test_process_line_valid_json(self):
        """Test processing a valid JSON line."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        json_line = '{"title": "Test", "content": "This is test content"}'
        doc = provider._process_line(json_line, 1, "/test/path")
        
        assert doc is not None
        assert doc.text == json_line  # Full JSON as text when text_field is None
        assert doc.metadata["line_number"] == 1
        assert doc.metadata["source_path"] == "/test/path"
        assert doc.metadata["json_title"] == "Test"
        assert doc.metadata["json_content"] == "This is test content"

    def test_process_line_with_text_field(self):
        """Test processing JSON line with specific text field."""
        config = StreamingJSONLReaderConfig(text_field="content")
        provider = StreamingJSONLReaderProvider(config)
        
        json_line = '{"title": "Test", "content": "This is test content"}'
        doc = provider._process_line(json_line, 1, "/test/path")
        
        assert doc is not None
        assert doc.text == "This is test content"
        assert doc.metadata["json_title"] == "Test"
        # Content field should not be duplicated in metadata
        assert "json_content" not in doc.metadata

    def test_process_line_empty_line(self):
        """Test that empty lines are skipped."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        assert provider._process_line("", 1, "/test/path") is None
        assert provider._process_line("   ", 1, "/test/path") is None
        assert provider._process_line("\n", 1, "/test/path") is None

    def test_process_line_malformed_json_strict_mode(self):
        """Test malformed JSON handling in strict mode."""
        config = StreamingJSONLReaderConfig(strict_mode=True)
        provider = StreamingJSONLReaderProvider(config)
        
        malformed_json = '{"title": "Test", "content": invalid}'
        
        with pytest.raises(ValueError, match="Malformed JSON at line 1"):
            provider._process_line(malformed_json, 1, "/test/path")

    def test_process_line_malformed_json_lenient_mode(self):
        """Test malformed JSON handling in lenient mode."""
        config = StreamingJSONLReaderConfig(strict_mode=False)
        provider = StreamingJSONLReaderProvider(config)
        
        malformed_json = '{"title": "Test", "content": invalid}'
        
        with patch('graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider.logger') as mock_logger:
            doc = provider._process_line(malformed_json, 1, "/test/path")
            assert doc is None
            mock_logger.warning.assert_called_once()

    def test_process_line_missing_text_field_strict_mode(self):
        """Test missing text field in strict mode."""
        config = StreamingJSONLReaderConfig(text_field="missing_field", strict_mode=True)
        provider = StreamingJSONLReaderProvider(config)
        
        json_line = '{"title": "Test", "content": "This is test content"}'
        
        with pytest.raises(ValueError, match="Missing text_field 'missing_field' at line 1"):
            provider._process_line(json_line, 1, "/test/path")

    def test_process_line_missing_text_field_lenient_mode(self):
        """Test missing text field in lenient mode."""
        config = StreamingJSONLReaderConfig(text_field="missing_field", strict_mode=False)
        provider = StreamingJSONLReaderProvider(config)
        
        json_line = '{"title": "Test", "content": "This is test content"}'
        
        with patch('graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider.logger') as mock_logger:
            doc = provider._process_line(json_line, 1, "/test/path")
            assert doc is None
            mock_logger.warning.assert_called_once()

    def test_process_line_empty_text_strict_mode(self):
        """Test empty text content in strict mode."""
        config = StreamingJSONLReaderConfig(text_field="content", strict_mode=True)
        provider = StreamingJSONLReaderProvider(config)
        
        json_line = '{"title": "Test", "content": ""}'
        
        with pytest.raises(ValueError, match="Empty text content at line 1"):
            provider._process_line(json_line, 1, "/test/path")

    def test_process_line_empty_text_lenient_mode(self):
        """Test empty text content in lenient mode."""
        config = StreamingJSONLReaderConfig(text_field="content", strict_mode=False)
        provider = StreamingJSONLReaderProvider(config)
        
        json_line = '{"title": "Test", "content": ""}'
        
        with patch('graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider.logger') as mock_logger:
            doc = provider._process_line(json_line, 1, "/test/path")
            assert doc is None
            mock_logger.warning.assert_called_once()

    def test_build_metadata_basic(self):
        """Test basic metadata building."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        json_obj = {"title": "Test", "content": "Content"}
        metadata = provider._build_metadata("/test/path", 5, json_obj)
        
        assert metadata["source_path"] == "/test/path"
        assert metadata["line_number"] == 5
        assert metadata["reader_type"] == "streaming_jsonl"
        assert metadata["json_title"] == "Test"
        assert metadata["json_content"] == "Content"

    def test_build_metadata_with_custom_function(self):
        """Test metadata building with custom function."""
        def custom_metadata(path):
            return {"custom_field": "custom_value", "path_length": len(path)}
        
        config = StreamingJSONLReaderConfig(metadata_fn=custom_metadata)
        provider = StreamingJSONLReaderProvider(config)
        
        json_obj = {"title": "Test"}
        metadata = provider._build_metadata("/test/path", 1, json_obj)
        
        assert metadata["custom_field"] == "custom_value"
        assert metadata["path_length"] == len("/test/path")
        assert metadata["json_title"] == "Test"

    def test_build_metadata_custom_function_exception(self):
        """Test metadata building when custom function raises exception."""
        def failing_metadata(path):
            raise Exception("Custom function failed")
        
        config = StreamingJSONLReaderConfig(metadata_fn=failing_metadata)
        provider = StreamingJSONLReaderProvider(config)
        
        json_obj = {"title": "Test"}
        
        with patch('graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider.logger') as mock_logger:
            metadata = provider._build_metadata("/test/path", 1, json_obj)
            # Should still work, just log warning
            assert metadata["json_title"] == "Test"
            mock_logger.warning.assert_called_once()

    def test_build_metadata_complex_json_objects(self):
        """Test metadata building with complex JSON objects."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        json_obj = {
            "title": "Test",
            "tags": ["tag1", "tag2"],
            "nested": {"key": "value"},
            "number": 42
        }
        metadata = provider._build_metadata("/test/path", 1, json_obj)
        
        assert metadata["json_title"] == "Test"
        assert metadata["json_tags"] == '["tag1", "tag2"]'
        assert metadata["json_nested"] == '{"key": "value"}'
        assert metadata["json_number"] == "42"

    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create temporary JSONL file
        test_data = [
            {"id": 1, "content": "First document"},
            {"id": 2, "content": "Second document"},
            {"id": 3, "content": "Third document"},
            {"id": 4, "content": "Fourth document"},
            {"id": 5, "content": "Fifth document"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(batch_size=2, text_field="content")
            provider = StreamingJSONLReaderProvider(config)
            
            batches = list(provider.lazy_load_data(temp_file))
            
            # Should have 3 batches: [2, 2, 1]
            assert len(batches) == 3
            assert len(batches[0]) == 2
            assert len(batches[1]) == 2
            assert len(batches[2]) == 1
            
            # Check content
            all_docs = []
            for batch in batches:
                all_docs.extend(batch)
            
            assert len(all_docs) == 5
            assert all_docs[0].text == "First document"
            assert all_docs[4].text == "Fifth document"
            
        finally:
            os.unlink(temp_file)

    def test_load_data_vs_lazy_load_data(self):
        """Test that load_data and lazy_load_data produce same results."""
        test_data = [
            {"id": 1, "content": "First document"},
            {"id": 2, "content": "Second document"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(text_field="content")
            provider = StreamingJSONLReaderProvider(config)
            
            # Load all at once
            all_docs = provider.load_data(temp_file)
            
            # Load in batches and combine
            lazy_docs = []
            for batch in provider.lazy_load_data(temp_file):
                lazy_docs.extend(batch)
            
            assert len(all_docs) == len(lazy_docs)
            for i in range(len(all_docs)):
                assert all_docs[i].text == lazy_docs[i].text
                assert all_docs[i].metadata["line_number"] == lazy_docs[i].metadata["line_number"]
                
        finally:
            os.unlink(temp_file)

    @patch('graphrag_toolkit.lexical_graph.indexing.load.readers.providers.streaming_jsonl_reader_provider.logger')
    def test_progress_logging(self, mock_logger):
        """Test progress logging functionality."""
        # Create file with enough lines to trigger logging
        test_data = [{"id": i, "content": f"Document {i}"} for i in range(15000)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            config = StreamingJSONLReaderConfig(log_interval=10000)
            provider = StreamingJSONLReaderProvider(config)
            
            # Process the file
            list(provider.lazy_load_data(temp_file))
            
            # Check that progress was logged
            info_calls = [call for call in mock_logger.info.call_args_list 
                         if "Processed" in str(call) and "lines" in str(call)]
            assert len(info_calls) >= 1  # Should have at least one progress log
            
        finally:
            os.unlink(temp_file)

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        with pytest.raises(FileNotFoundError):
            list(provider.lazy_load_data("/non/existent/file.jsonl"))

    @patch.object(StreamingJSONLReaderProvider, '_process_file_paths')
    def test_s3_integration(self, mock_process_paths):
        """Test S3 integration through S3FileMixin."""
        # Create a temporary file to simulate downloaded S3 file
        test_data = [{"id": 1, "content": "S3 document"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            # Mock the S3 file processing to return the actual temp file
            mock_process_paths.return_value = ([temp_file], [temp_file], ["s3://bucket/file.jsonl"])
            
            config = StreamingJSONLReaderConfig(text_field="content")
            provider = StreamingJSONLReaderProvider(config)
            
            docs = provider.load_data("s3://bucket/file.jsonl")
            
            assert len(docs) == 1
            assert docs[0].text == "S3 document"
            assert docs[0].metadata["source_path"] == "s3://bucket/file.jsonl"
            
            # Verify S3 processing was called
            mock_process_paths.assert_called_once_with("s3://bucket/file.jsonl")
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_error_handling_during_streaming(self):
        """Test error handling during file streaming."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        # Mock _process_file_paths to return valid paths, then mock file opening to fail
        with patch.object(provider, '_process_file_paths', return_value=(["/test/file.jsonl"], [], ["/test/file.jsonl"])):
            with patch('builtins.open', side_effect=Exception("File read failed")):
                with pytest.raises(RuntimeError, match="Failed to stream JSONL"):
                    list(provider.lazy_load_data("/test/file.jsonl"))

    def test_cleanup_on_exception(self):
        """Test that temporary files are cleaned up on exception."""
        config = StreamingJSONLReaderConfig()
        provider = StreamingJSONLReaderProvider(config)
        
        mock_cleanup = MagicMock()
        
        with patch.object(provider, '_process_file_paths', return_value=([], ["temp_file"], [])):
            with patch.object(provider, '_cleanup_temp_files', mock_cleanup):
                with patch('builtins.open', side_effect=Exception("File read failed")):
                    with pytest.raises(RuntimeError):
                        list(provider.lazy_load_data("/test/file.jsonl"))
                    
                    # Verify cleanup was called
                    mock_cleanup.assert_called_once_with(["temp_file"])