# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from unittest.mock import patch
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import YouTubeReaderConfig
from graphrag_toolkit.lexical_graph.indexing.load.readers.providers.youtube_reader_provider import YouTubeReaderProvider


class TestYouTubeReaderProviderProxyConfig:
    """Test cases for YouTube provider proxy configuration."""

    def test_proxy_url_from_config(self):
        """Test proxy URL is taken from config when provided."""
        config = YouTubeReaderConfig(
            language="en",
            proxy_url="http://user:pass@proxy.example.com:8080"
        )
        
        with patch.dict(os.environ, {}, clear=True):
            provider = YouTubeReaderProvider(config)
            assert provider.proxy_url == "http://user:pass@proxy.example.com:8080"

    def test_proxy_url_from_env_var(self):
        """Test proxy URL falls back to environment variable when not in config."""
        config = YouTubeReaderConfig(language="en")
        
        with patch.dict(os.environ, {'YOUTUBE_PROXY_URL': 'http://env.proxy.com:3128'}, clear=True):
            provider = YouTubeReaderProvider(config)
            assert provider.proxy_url == "http://env.proxy.com:3128"

    def test_config_proxy_overrides_env_var(self):
        """Test config proxy URL takes precedence over environment variable."""
        config = YouTubeReaderConfig(
            language="en",
            proxy_url="http://config.proxy.com:8080"
        )
        
        with patch.dict(os.environ, {'YOUTUBE_PROXY_URL': 'http://env.proxy.com:3128'}, clear=True):
            provider = YouTubeReaderProvider(config)
            assert provider.proxy_url == "http://config.proxy.com:8080"

    def test_no_proxy_configuration(self):
        """Test provider works without proxy configuration."""
        config = YouTubeReaderConfig(language="en")
        
        with patch.dict(os.environ, {}, clear=True):
            provider = YouTubeReaderProvider(config)
            assert provider.proxy_url is None

    def test_proxy_url_validation_in_config(self):
        """Test that proxy URL can be None in config."""
        config = YouTubeReaderConfig(
            language="en",
            proxy_url=None
        )
        
        with patch.dict(os.environ, {}, clear=True):
            provider = YouTubeReaderProvider(config)
            assert provider.proxy_url is None

    def test_empty_proxy_url_in_config(self):
        """Test that empty string proxy URL falls back to environment variable."""
        config = YouTubeReaderConfig(
            language="en",
            proxy_url=""  # Empty string should be falsy
        )
        
        with patch.dict(os.environ, {'YOUTUBE_PROXY_URL': 'http://env.proxy.com:3128'}, clear=True):
            provider = YouTubeReaderProvider(config)
            assert provider.proxy_url == "http://env.proxy.com:3128"

    def test_config_attributes_set_correctly(self):
        """Test that all config attributes are set correctly."""
        config = YouTubeReaderConfig(
            language="es",
            proxy_url="http://test.proxy.com:8080"
        )
        
        provider = YouTubeReaderProvider(config)
        assert provider.language == "es"
        assert provider.proxy_url == "http://test.proxy.com:8080"
        assert provider.metadata_fn is None

    def test_environment_vs_programmatic_setting(self):
        """Test that programmatic setting overrides environment variables."""
        config = YouTubeReaderConfig(language="en")
        
        with patch.dict(os.environ, {'YOUTUBE_PROXY_URL': 'http://env.proxy.com:3128'}, clear=True):
            # First provider should use environment
            provider1 = YouTubeReaderProvider(config)
            assert provider1.proxy_url == "http://env.proxy.com:3128"
            
            # Second provider with explicit config should override
            config_with_proxy = YouTubeReaderConfig(
                language="en",
                proxy_url="http://explicit.proxy.com:8080"
            )
            provider2 = YouTubeReaderProvider(config_with_proxy)
            assert provider2.proxy_url == "http://explicit.proxy.com:8080"