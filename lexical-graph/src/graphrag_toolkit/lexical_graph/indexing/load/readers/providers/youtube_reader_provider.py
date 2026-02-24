from typing import List, Union
import re
from llama_index.core.schema import Document
from ..reader_provider_config import YouTubeReaderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class YouTubeReaderProvider:
    """Direct YouTube transcript reader using youtube-transcript-api."""

    def __init__(self, config: YouTubeReaderConfig):
        self.language = config.language
        self.metadata_fn = config.metadata_fn
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
        """Read YouTube transcript documents."""
        if not input_source:
            logger.error("No input source provided to YouTubeReaderProvider")
            raise ValueError("input_source cannot be None or empty")
        
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as e:
            logger.error("Failed to import YouTubeTranscriptApi: missing youtube-transcript-api")
            raise ImportError(
                "YouTubeTranscriptApi requires 'youtube-transcript-api'. "
                "Install with: pip install youtube-transcript-api"
            ) from e

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