# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import Nova2MultimodalEmbedding


class TestNova2MultimodalEmbedding:
    """Test suite for Nova2MultimodalEmbedding class."""
    
    def test_initialization(self):
        """Test basic initialization of Nova2MultimodalEmbedding."""
        embedding = Nova2MultimodalEmbedding(
            model_name="amazon.nova-2-multimodal-embeddings-v1:0",
            embed_dimensions=3072,
            embed_purpose="TEXT_RETRIEVAL",
            truncation_mode="END"
        )
        
        assert embedding.model_name == "amazon.nova-2-multimodal-embeddings-v1:0"
        assert embedding.embed_dimensions == 3072
        assert embedding.embed_purpose == "TEXT_RETRIEVAL"
        assert embedding.truncation_mode == "END"
        assert embedding._client is None
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        assert embedding.embed_dimensions == 3072
        assert embedding.embed_purpose == "TEXT_RETRIEVAL"
        assert embedding.truncation_mode == "END"
    
    def test_build_request_body(self):
        """Test the Nova 2 API request body format."""
        embedding = Nova2MultimodalEmbedding(
            model_name="amazon.nova-2-multimodal-embeddings-v1:0",
            embed_dimensions=1024,
            embed_purpose="CLASSIFICATION",
            truncation_mode="NONE"
        )
        
        request_body = embedding._build_request_body("test text")
        
        expected = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingDimension": 1024,
                "embeddingPurpose": "CLASSIFICATION",
                "text": {
                    "truncationMode": "NONE",
                    "value": "test text"
                }
            }
        }
        
        assert request_body == expected
    
    def test_empty_text_validation(self):
        """Test that empty or whitespace-only text raises ValueError."""
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        # Test empty string
        with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only"):
            embedding._get_embedding("")
        
        # Test whitespace-only string
        with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only"):
            embedding._get_embedding("   \n\t   ")
    
    @patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig')
    def test_client_lazy_initialization(self, mock_config):
        """Test that the bedrock client is lazily initialized."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_config.session = mock_session
        
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        # Client should be None initially
        assert embedding._client is None
        
        # Accessing client property should initialize it
        client = embedding.client
        assert client == mock_client
        mock_session.client.assert_called_once_with('bedrock-runtime')
        
        # Subsequent access should return cached client
        client2 = embedding.client
        assert client2 == mock_client
        assert mock_session.client.call_count == 1
    
    @patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig')
    def test_successful_embedding_call(self, mock_config):
        """Test successful embedding API call."""
        # Setup mocks
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_config.session = mock_session
        
        # Mock successful API response
        mock_response = {
            'body': Mock()
        }
        mock_response_body = {
            'embeddings': [
                {
                    'embeddingType': 'TEXT',
                    'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            ]
        }
        mock_response['body'].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        result = embedding._get_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Verify API call
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]['modelId'] == "amazon.nova-2-multimodal-embeddings-v1:0"
        assert call_args[1]['contentType'] == "application/json"
        assert call_args[1]['accept'] == "application/json"
        
        # Verify request body
        request_body = json.loads(call_args[1]['body'])
        assert request_body['taskType'] == "SINGLE_EMBEDDING"
        assert request_body['singleEmbeddingParams']['text']['value'] == "test text"
    
    def test_retryable_error_detection(self):
        """Test detection of retryable errors."""
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        # Test retryable error types
        retryable_errors = [
            Exception("ModelErrorException occurred"),
            Exception("ThrottlingException: Rate limit exceeded"),
            Exception("ServiceUnavailableException"),
            Exception("unexpected error during processing"),
            Exception("try your request again"),
            Exception("service unavailable"),
            Exception("throttling detected")
        ]
        
        for error in retryable_errors:
            assert embedding._is_retryable_error(error) is True
        
        # Test non-retryable errors
        non_retryable_errors = [
            Exception("ValidationException: Invalid input"),
            Exception("AccessDeniedException: Insufficient permissions"),
            Exception("ResourceNotFoundException: Model not found")
        ]
        
        for error in non_retryable_errors:
            assert embedding._is_retryable_error(error) is False
    
    @patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig')
    @patch('time.sleep')
    def test_retry_logic_with_transient_error(self, mock_sleep, mock_config):
        """Test retry logic with transient errors."""
        # Setup mocks
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_config.session = mock_session
        
        # Mock transient error followed by success
        mock_response = {
            'body': Mock()
        }
        mock_response_body = {
            'embeddings': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        mock_response['body'].read.return_value = json.dumps(mock_response_body).encode()
        
        mock_client.invoke_model.side_effect = [
            Exception("ThrottlingException: Rate limit exceeded"),
            mock_response
        ]
        
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        result = embedding._get_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_client.invoke_model.call_count == 2
        mock_sleep.assert_called_once()  # Should have slept once for retry
    
    @patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig')
    def test_non_retryable_error_immediate_failure(self, mock_config):
        """Test that non-retryable errors fail immediately."""
        # Setup mocks
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_config.session = mock_session
        
        # Mock non-retryable error
        mock_client.invoke_model.side_effect = Exception("ValidationException: Invalid input")
        
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        with pytest.raises(Exception, match="ValidationException: Invalid input"):
            embedding._get_embedding("test text")
        
        # Should only try once for non-retryable error
        assert mock_client.invoke_model.call_count == 1
    
    @patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig')
    @patch('time.sleep')
    def test_max_retries_exceeded(self, mock_sleep, mock_config):
        """Test behavior when max retries are exceeded."""
        # Setup mocks
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_config.session = mock_session
        
        # Mock persistent retryable error
        mock_client.invoke_model.side_effect = Exception("ThrottlingException: Rate limit exceeded")
        
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        with pytest.raises(Exception, match="ThrottlingException: Rate limit exceeded"):
            embedding._get_embedding("test text")
        
        # Should try MAX_RETRIES times
        assert mock_client.invoke_model.call_count == 5  # MAX_RETRIES = 5
        assert mock_sleep.call_count == 4  # Sleep between retries (not after last attempt)
    
    def test_pickle_support(self):
        """Test pickle/unpickle support for multiprocessing."""
        import pickle
        
        embedding = Nova2MultimodalEmbedding(
            model_name="amazon.nova-2-multimodal-embeddings-v1:0",
            embed_dimensions=1024
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(embedding)
        unpickled = pickle.loads(pickled)
        
        # Verify attributes are preserved
        assert unpickled.model_name == embedding.model_name
        assert unpickled.embed_dimensions == embedding.embed_dimensions
        assert unpickled.embed_purpose == embedding.embed_purpose
        assert unpickled.truncation_mode == embedding.truncation_mode
        
        # Client should be None after unpickling
        assert unpickled._client is None
    
    def test_interface_methods(self):
        """Test that all required BaseEmbedding interface methods are implemented."""
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        
        # Test that methods exist and are callable
        assert callable(embedding._get_text_embedding)
        assert callable(embedding._get_query_embedding)
        assert callable(embedding._aget_text_embedding)
        assert callable(embedding._aget_query_embedding)
        
        # Test class name
        assert embedding.class_name() == "Nova2MultimodalEmbedding"
    
    @patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig')
    def test_empty_embeddings_response(self, mock_config):
        """Test handling of empty embeddings in API response."""
        # Setup mocks
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_config.session = mock_session
        
        # Mock response with empty embeddings
        mock_response = {
            'body': Mock()
        }
        mock_response_body = {'embeddings': []}
        mock_response['body'].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        embedding = Nova2MultimodalEmbedding("amazon.nova-2-multimodal-embeddings-v1:0")
        result = embedding._get_embedding("test text")
        
        assert result == []