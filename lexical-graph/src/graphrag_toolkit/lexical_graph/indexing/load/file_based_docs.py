# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from datetime import datetime
from os.path import join
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.model import (
    SourceDocument,
    SourceType,
    source_documents_from_source_types,
)
from graphrag_toolkit.lexical_graph.indexing.constants import (
    PROPOSITIONS_KEY,
    TOPICS_KEY,
)
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)


class FileBasedDocs(NodeHandler):
    """
    Handles reading, processing, and organizing documents from a file-based
    directory structure.

    This class is used to manage a directory containing documents and their
    associated metadata. It provides methods for filtering metadata, iterating
    through the directory structure to yield processed documents, and handling
    document storage. It ensures compatibility with metadata filtering and
    document organization needs.

    :ivar docs_directory: The base directory where documents are stored.
    :type docs_directory: str
    :ivar collection_id: Identifier for the collection within the documents directory.
    :type collection_id: str
    :ivar metadata_keys: Optional list of metadata keys to filter during processing.
    :type metadata_keys: Optional[List[str]]
    """

    docs_directory: str
    collection_id: str

    metadata_keys: Optional[List[str]]

    def __init__(
        self,
        docs_directory: str,
        collection_id: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
    ):
        """
        Initializes an instance of the class and sets up the required directory for storing
        documents associated with the collection. The initialization also generates a unique
        collection ID if not provided, and configures metadata keys required for processing.

        :param docs_directory: The base directory where the documents will be stored.
        :type docs_directory: str
        :param collection_id: An optional identifier for the document collection. If not
            provided, a unique ID based on the current timestamp will be generated.
        :type collection_id: Optional[str]
        :param metadata_keys: List of metadata keys for additional information associated
            with the documents. Defaults to None.
        :type metadata_keys: Optional[List[str]]
        """
        super().__init__(
            docs_directory=docs_directory,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            metadata_keys=metadata_keys,
        )
        self._prepare_directory(join(self.docs_directory, self.collection_id))

    def _prepare_directory(self, directory_path):
        """
        Creates a directory if it does not already exist.

        This function checks whether the specified directory path exists. If the directory
        does not exist, it creates it using the appropriate system calls.

        :param directory_path: Path to the directory that needs to be prepared.
        :type directory_path: str
        :return: None
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def docs(self):
        """
        Provides methods and attributes for handling objects.

        This class is used to perform operations and return the same object
        when requested.

        :Attributes:
            self (object): The instance of the class itself.

        :Methods:
            docs:
                Returns the instance of the class.

        :return: The class instance (`self`).
        :rtype: object
        """
        return self

    def _filter_metadata(self, node: TextNode) -> TextNode:
        """
        Filters metadata of a TextNode object by removing undesired keys. The method ensures
        that only selected keys are retained in the metadata of both the node itself and its
        relationships.

        The filtering mechanism is aligned with either predefined keys or a list of metadata
        keys specified by the class. If a key is not in the allowed list, it will be removed
        from the metadata.

        :param node: A TextNode object whose metadata needs to be filtered.
        :type node: TextNode
        :return: The filtered TextNode object with updated metadata.
        :rtype: TextNode
        """

        def filter(metadata: Dict):
            """
            Handles filtering and processing of metadata for text nodes in a file-based system.
            This class extends `NodeHandler` and modifies metadata information for specific
            text nodes based on allowed keys and a set of predefined filtering rules.

            The focus of the class is to ensure that metadata attached to text nodes is
            appropriately filtered, retaining only the relevant and necessary information.

            :param metadata_keys: List of metadata keys allowed for processing.
            :type metadata_keys: Optional[List[str]]
            """
            keys_to_delete = []
            for key in metadata.keys():
                if key not in [PROPOSITIONS_KEY, TOPICS_KEY, INDEX_KEY]:
                    if self.metadata_keys is not None and key not in self.metadata_keys:
                        keys_to_delete.append(key)
            for key in keys_to_delete:
                del metadata[key]

        filter(node.metadata)

        for _, relationship_info in node.relationships.items():
            if relationship_info.metadata:
                filter(relationship_info.metadata)

        return node

    def __iter__(self):
        """
        Iterates over the source document directories within the specified collection directory,
        reads files, filters metadata, and yields `SourceDocument` instances containing filtered
        text nodes. The method accesses the target directory based on the `docs_directory` and
        `collection_id` attributes and processes each subdirectory as a source document directory.

        :raises FileNotFoundError: A `FileNotFoundError` may be raised if the `docs_directory` or
            specific collection directory does not exist.
        :raises OSError: May occur if there are issues while accessing the file system or reading files.

        :yields: Instances of `SourceDocument` containing text node data processed from source
            document directories.
        :rtype: Iterator[SourceDocument]
        """
        directory_path = join(self.docs_directory, self.collection_id)

        logger.debug(f'Reading source documents from directory: {directory_path}')

        source_document_directory_paths = [
            f.path for f in os.scandir(directory_path) if f.is_dir()
        ]

        for source_document_directory_path in source_document_directory_paths:
            nodes = []
            for filename in os.listdir(source_document_directory_path):
                file_path = os.path.join(source_document_directory_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path) as f:
                        nodes.append(
                            self._filter_metadata(TextNode.from_json(f.read()))
                        )
            yield SourceDocument(nodes=nodes)

    def __call__(self, nodes: List[SourceType], **kwargs: Any) -> List[SourceDocument]:
        """
        Makes the class instance callable and processes a list of `SourceType` nodes
        to generate a list of `SourceDocument` objects. The processing involves
        extracting source documents from the source types provided in the input
        and applying additional operations using the given keyword arguments.

        :param nodes: A list of source type instances to be processed.
        :type nodes: List[SourceType]
        :param kwargs: Additional keyword arguments for processing the nodes.
        :type kwargs: Any
        :return: A list of processed source documents derived from the input
            source types.
        :rtype: List[SourceDocument]
        """
        return [
            n for n in self.accept(source_documents_from_source_types(nodes), **kwargs)
        ]

    def accept(
        self, source_documents: List[SourceDocument], **kwargs: Any
    ) -> Generator[SourceDocument, None, None]:
        """
        Processes and writes source documents to specified directories, preparing directories,
        logging actions, and yielding the document objects after processing.

        :param source_documents: List of SourceDocument instances to be processed. Each document
            contains nodes which will be saved as separate JSON files.
        :type source_documents: List[SourceDocument]
        :param kwargs: Additional keyword arguments that may customize the behavior of the function.
        :type kwargs: Any
        :return: A generator yielding each SourceDocument after it has been processed and saved.
        :rtype: Generator[SourceDocument, None, None]
        """
        for source_document in source_documents:
            directory_path = join(
                self.docs_directory, self.collection_id, source_document.source_id()
            )
            self._prepare_directory(directory_path)
            logger.debug(f'Writing source document to directory: {directory_path}')
            for node in source_document.nodes:
                if not [key for key in [INDEX_KEY] if key in node.metadata]:
                    chunk_output_path = join(directory_path, f'{node.node_id}.json')
                    logger.debug(f'Writing chunk to file: {chunk_output_path}')
                    with open(chunk_output_path, 'w') as f:
                        json.dump(node.to_dict(), f, indent=4)
            yield source_document
