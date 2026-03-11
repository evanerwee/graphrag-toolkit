# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from unittest.mock import patch
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig


class TestConfigOutputDirectories:
    """Test cases for output directory configuration options."""

    def test_local_output_dir_default(self):
        """Test local_output_dir returns default value."""
        # Reset the config
        GraphRAGConfig._local_output_dir = None
        
        with patch.dict(os.environ, {}, clear=True):
            assert GraphRAGConfig.local_output_dir == 'output'

    def test_local_output_dir_from_env(self):
        """Test local_output_dir reads from environment variable."""
        # Reset the config
        GraphRAGConfig._local_output_dir = None
        
        with patch.dict(os.environ, {'LOCAL_OUTPUT_DIR': '/tmp/custom'}, clear=True):
            assert GraphRAGConfig.local_output_dir == '/tmp/custom'

    def test_local_output_dir_setter(self):
        """Test local_output_dir can be set programmatically."""
        GraphRAGConfig.local_output_dir = '/custom/path'
        assert GraphRAGConfig.local_output_dir == '/custom/path'
        
        # Reset for other tests
        GraphRAGConfig._local_output_dir = None

    def test_log_output_dir_default(self):
        """Test log_output_dir returns default value (None)."""
        # Reset the config
        GraphRAGConfig._log_output_dir = None
        
        with patch.dict(os.environ, {}, clear=True):
            assert GraphRAGConfig.log_output_dir is None

    def test_log_output_dir_from_env(self):
        """Test log_output_dir reads from environment variable."""
        # Reset the config
        GraphRAGConfig._log_output_dir = None
        
        with patch.dict(os.environ, {'LOG_OUTPUT_DIR': '/tmp/logs'}, clear=True):
            assert GraphRAGConfig.log_output_dir == '/tmp/logs'

    def test_log_output_dir_setter(self):
        """Test log_output_dir can be set programmatically."""
        GraphRAGConfig.log_output_dir = '/custom/logs'
        assert GraphRAGConfig.log_output_dir == '/custom/logs'
        
        # Reset for other tests
        GraphRAGConfig._log_output_dir = None

    def test_youtube_proxy_url_default(self):
        """Test youtube_proxy_url returns default value (None)."""
        # Reset the config
        GraphRAGConfig._youtube_proxy_url = None
        
        with patch.dict(os.environ, {}, clear=True):
            assert GraphRAGConfig.youtube_proxy_url is None

    def test_youtube_proxy_url_from_env(self):
        """Test youtube_proxy_url reads from environment variable."""
        # Reset the config
        GraphRAGConfig._youtube_proxy_url = None
        
        proxy_url = 'http://user:pass@proxy.example.com:8080'
        with patch.dict(os.environ, {'YOUTUBE_PROXY_URL': proxy_url}, clear=True):
            assert GraphRAGConfig.youtube_proxy_url == proxy_url

    def test_youtube_proxy_url_setter(self):
        """Test youtube_proxy_url can be set programmatically."""
        proxy_url = 'http://user:pass@proxy.example.com:8080'
        GraphRAGConfig.youtube_proxy_url = proxy_url
        assert GraphRAGConfig.youtube_proxy_url == proxy_url
        
        # Reset for other tests
        GraphRAGConfig._youtube_proxy_url = None

    def test_config_persistence(self):
        """Test that config values persist across multiple accesses."""
        GraphRAGConfig.local_output_dir = '/persistent/path'
        GraphRAGConfig.log_output_dir = '/persistent/logs'
        
        # Access multiple times to ensure persistence
        assert GraphRAGConfig.local_output_dir == '/persistent/path'
        assert GraphRAGConfig.log_output_dir == '/persistent/logs'
        assert GraphRAGConfig.local_output_dir == '/persistent/path'
        
        # Reset for other tests
        GraphRAGConfig._local_output_dir = None
        GraphRAGConfig._log_output_dir = None

    def test_environment_vs_programmatic_setting(self):
        """Test that programmatic setting overrides environment variables."""
        # Reset the config
        GraphRAGConfig._local_output_dir = None
        
        with patch.dict(os.environ, {'LOCAL_OUTPUT_DIR': '/env/path'}, clear=True):
            # First access should use environment
            assert GraphRAGConfig.local_output_dir == '/env/path'
            
            # Programmatic setting should override
            GraphRAGConfig.local_output_dir = '/programmatic/path'
            assert GraphRAGConfig.local_output_dir == '/programmatic/path'
        
        # Reset for other tests
        GraphRAGConfig._local_output_dir = None