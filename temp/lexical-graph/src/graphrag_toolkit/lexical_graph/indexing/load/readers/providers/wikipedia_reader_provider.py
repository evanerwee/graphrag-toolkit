from typing import List, Union
from llama_index.core.schema import Document
from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import WikipediaReaderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class WikipediaReaderProvider:
    """Reader provider for Wikipedia articles using LlamaIndex's WikipediaReader."""

    def __init__(self, config: WikipediaReaderConfig):
        self.config = config
        self.lang = config.lang
        self.metadata_fn = config.metadata_fn
        self._reader = None
        logger.debug(f"Initialized WikipediaReaderProvider with lang={config.lang}")

    def _init_reader(self):
        """Lazily initialize WikipediaReader if not already created."""
        if self._reader is None:
            try:
                from llama_index.readers.wikipedia import WikipediaReader
            except ImportError as e:
                logger.error("Failed to import WikipediaReader: missing wikipedia package")
                raise ImportError(
                    "WikipediaReader requires the 'wikipedia' package. Install with: pip install wikipedia"
                ) from e
            self._reader = WikipediaReader()

    def read(self, input_source: Union[str, List[str]]) -> List[Document]:
        """Read Wikipedia documents with metadata handling and title correction."""
        # DEBUG: Log exactly what we received
        logger.info(f"🔍 DEBUG WikipediaReaderProvider.read() called with: {repr(input_source)}")
        logger.info(f"🔍 DEBUG input_source type: {type(input_source)}")
        if isinstance(input_source, str):
            logger.info(f"🔍 DEBUG input_source bytes: {input_source.encode('utf-8')}")
        
        if not input_source:
            logger.error("No input source provided to WikipediaReaderProvider")
            raise ValueError("input_source cannot be None or empty")
        
        self._init_reader()

        try:
            import wikipedia
            logger.info(f"🔍 DEBUG wikipedia package version: {wikipedia.__version__ if hasattr(wikipedia, '__version__') else 'unknown'}")
        except ImportError as e:
            logger.error("Failed to import wikipedia package")
            raise ImportError(
                "The 'wikipedia' package is required for WikipediaReaderProvider. Install it with: pip install wikipedia"
            ) from e

        pages = [input_source] if isinstance(input_source, str) else input_source
        logger.info(f"Reading {len(pages)} Wikipedia page(s)")
        logger.info(f"🔍 DEBUG pages list: {repr(pages)}")
        validated_pages = []
        
        for page in pages:
            # Convert underscores to spaces (Wikipedia URLs use underscores, API uses spaces)
            page_title = page.replace('_', ' ')
            logger.info(f"🔍 DEBUG Processing page: {repr(page)} -> page_title: {repr(page_title)}")
            try:
                wikipedia.set_lang(self.lang)
                # CRITICAL: auto_suggest=False prevents the library from mangling titles
                # e.g., "Computer security" -> "computers security" (wrong!)
                logger.info(f"🔍 DEBUG Calling wikipedia.page({repr(page_title)}, auto_suggest=False)")
                wiki_page = wikipedia.page(page_title, auto_suggest=False)
                logger.info(f"🔍 DEBUG wikipedia.page() returned: title={repr(wiki_page.title)}, pageid={wiki_page.pageid}")
                validated_pages.append(page_title)
                logger.debug(f"Validated Wikipedia page: {page_title}")
            except wikipedia.exceptions.PageError as e:
                logger.warning(f"🔍 DEBUG PageError for {repr(page_title)}: {e}")
                try:
                    if search_results := wikipedia.search(page_title, results=1):
                        logger.info(f"🔍 DEBUG search results: {search_results}")
                        wikipedia.page(search_results[0], auto_suggest=False)
                        validated_pages.append(search_results[0])
                        logger.info(f"Corrected page title: '{page_title}' -> '{search_results[0]}'")
                    else:
                        logger.warning(f"No Wikipedia page found for '{page_title}'")
                except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
                    logger.warning(f"Could not resolve Wikipedia page for '{page_title}': {e}")

        if not validated_pages:
            logger.error(f"No valid Wikipedia pages found for: {pages}")
            raise ValueError(f"No valid Wikipedia pages found for: {pages}")

        try:
            logger.info(f"🔍 DEBUG Calling self._reader.load_data(pages={repr(validated_pages)})")
            documents = self._reader.load_data(pages=validated_pages)
            logger.info(f"Successfully read {len(documents)} document(s) from Wikipedia")

            if self.metadata_fn:
                for doc in documents:
                    page_context = validated_pages[0] if validated_pages else str(input_source)
                    additional_metadata = self.metadata_fn(page_context)
                    doc.metadata.update(additional_metadata)

            return documents
        except Exception as e:
            logger.error(f"Failed to read Wikipedia pages {validated_pages}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to read Wikipedia pages: {e}") from e
