# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union
import re
import os
from llama_index.core.schema import Document
from ..reader_provider_config import YouTubeReaderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class YouTubeReaderProvider:
    """Direct YouTube transcript reader using youtube-transcript-api.
    
    Supports proxy configuration to bypass cloud provider IP blocks.
    YouTube blocks requests from AWS, GCP, and Azure IP ranges.
    """

    def __init__(self, config: YouTubeReaderConfig):
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as e:
            logger.error("Failed to import YouTubeTranscriptApi: missing youtube-transcript-api")
            raise ImportError(
                "youtube-transcript-api package not found, install with 'pip install youtube-transcript-api'"
            ) from e
        
        self.language = config.language
        self.metadata_fn = config.metadata_fn
        
        # Provider-level proxy configuration with environment variable fallback
        self.proxy_url = config.proxy_url or os.environ.get('YOUTUBE_PROXY_URL')
        
        if self.proxy_url:
            # Mask credentials in logs
            masked_proxy = self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url
            logger.info(f"YouTube proxy configured: {masked_proxy}")
        else:
            logger.debug("No YouTube proxy configured - may fail on cloud provider IPs")
            
        logger.debug(f"Initialized YouTubeReaderProvider with language={config.language}")

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        logger.error(f"Could not extract video ID from URL: {url}")
        raise ValueError(f"Could not extract video ID from URL: {url}")

    def read(self, input_source: Union[str, List[str]]) -> List[Document]:
        """Read YouTube transcript documents.
        
        Automatically configures proxy environment variables if proxy_url is set.
        This bypasses cloud provider IP blocks that YouTube enforces.
        """
        if not input_source:
            logger.error("No input source provided to YouTubeReaderProvider")
            raise ValueError("input_source cannot be None or empty")
        
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Store original proxy settings to restore later
        original_http_proxy = os.environ.get('HTTP_PROXY')
        original_https_proxy = os.environ.get('HTTPS_PROXY')
        
        # Set proxy environment variables if configured
        # youtube-transcript-api respects HTTP_PROXY/HTTPS_PROXY
        if self.proxy_url:
            logger.debug("Setting proxy environment variables for YouTube API")
            os.environ['HTTP_PROXY'] = self.proxy_url
            os.environ['HTTPS_PROXY'] = self.proxy_url

        try:
            urls = [input_source] if isinstance(input_source, str) else input_source
            logger.info(f"Reading transcripts from {len(urls)} YouTube video(s)")
            documents = []
            
            for url in urls:
                try:
                    video_id = self._extract_video_id(url)
                    logger.debug(f"Processing video ID: {video_id}")
                    
                    api = YouTubeTranscriptApi()
                    transcript_list = api.fetch(video_id, languages=[self.language])
                    
                    if isinstance(transcript_list, list):
                        full_text = " ".join([segment.get('text', '') for segment in transcript_list])
                    else:
                        full_text = str(transcript_list)
                    
                    metadata = {
                        'video_id': video_id,
                        'url': url,
                        'language': self.language,
                        'source': 'youtube'
                    }
                    
                    if self.metadata_fn:
                        custom_metadata = self.metadata_fn(url)
                        metadata.update(custom_metadata)
                    
                    doc = Document(text=full_text, metadata=metadata)
                    documents.append(doc)
                    logger.info(f"Successfully read transcript for video {video_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to read transcript for {url} with language {self.language}: {e}")
                    try:
                        api = YouTubeTranscriptApi()
                        transcript_list = api.fetch(video_id)
                        
                        if isinstance(transcript_list, list):
                            full_text = " ".join([segment.get('text', '') for segment in transcript_list])
                        else:
                            full_text = str(transcript_list)
                        
                        metadata = {
                            'video_id': video_id,
                            'url': url,
                            'language': 'auto',
                            'source': 'youtube'
                        }
                        
                        if self.metadata_fn:
                            custom_metadata = self.metadata_fn(url)
                            metadata.update(custom_metadata)
                        
                        doc = Document(text=full_text, metadata=metadata)
                        documents.append(doc)
                        logger.info(f"Successfully read transcript for video {video_id} with auto language")
                        
                    except Exception as e2:
                        logger.error(f"Failed to read transcript for {url} (fallback also failed): {e2}", exc_info=True)
                        continue
            
            logger.info(f"Successfully read {len(documents)} YouTube transcript(s)")
            return documents
            
        finally:
            # Restore original proxy settings
            if original_http_proxy is not None:
                os.environ['HTTP_PROXY'] = original_http_proxy
            elif 'HTTP_PROXY' in os.environ and self.proxy_url:
                del os.environ['HTTP_PROXY']
            
            if original_https_proxy is not None:
                os.environ['HTTPS_PROXY'] = original_https_proxy
            elif 'HTTPS_PROXY' in os.environ and self.proxy_url:
                del os.environ['HTTPS_PROXY']