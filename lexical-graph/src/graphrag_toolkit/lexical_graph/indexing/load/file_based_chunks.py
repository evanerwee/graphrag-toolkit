# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from datetime import datetime
from os.path import join
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.constants import (
    PROPOSITIONS_KEY,
    TOPICS_KEY,
)
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)


class FileBasedChunks(NodeHandler):
    """
    Handles file-based data chunking with support for metadata filtering and
    structured data organization.

    This class allows efficient storage and retrieval of data chunks from the
    filesystem. It provides methods for filtering metadata, reading files and
    yielding structured data objects, writing data to disk, and organizing files
    for specific collections. Primarily designed for scenarios requiring metadata
    management and chunk-based data processing.

    :ivar chunks_directory: The base directory for storing and organizing chunk files.
    :type chunks_directory: str
    :ivar collection_id: Identifier for the collection. If not provided during
        initialization, a timestamp in the format '%Y%m%d-%H%M%S' will be used.
    :type collection_id: str
    :ivar metadata_keys: List of metadata keys to include or filter during
        processing. If None, all metadata are included.
    :type metadata_keys: Optional[List[str]]
    """

    chunks_directory: str
    collection_id: str

    metadata_keys: Optional[List[str]]

    def __init__(
        self,
        chunks_directory: str,
        collection_id: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
    ):
        """
        Initializes an instance of the class responsible for handling a directory of chunks,
        a specific collection identifier, and optional metadata keys. The constructor sets
        up the necessary parameters and ensures the storage directory is prepared for
        further operations. A collection identifier can optionally be provided, but if not,
        it defaults to a timestamp.

        :param chunks_directory: The path to the directory where chunk files are stored.
            This parameter is mandatory and determines the location for reading or writing
            chunked data.
        :type chunks_directory: str
        :param collection_id: A unique identifier for the collection of chunks being
            handled. If not provided, a timestamp in the format "YYYYMMDD-HHMMSS" is
            automatically generated.
        :type collection_id: Optional[str]
        :param metadata_keys: A list of keys for associating metadata with collection items.
            These keys allow additional information to be attached to the handled objects.
        :type metadata_keys: Optional[List[str]]
        """
        super().__init__(
            chunks_directory=chunks_directory,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            metadata_keys=metadata_keys,
        )
        self._prepare_directory()

    def _prepare_directory(self):
        """
        Prepare the directory for storing collection-related files.

        This method ensures that the intended directory for storing files
        associated with a specific collection exists. If the directory does
        not exist, it is created.

        :return: None
        """
        directory_path = join(self.chunks_directory, self.collection_id)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def chunks(self):
        """
        Provides an iterable that returns itself. Typically used in cases where
        an iterator behavior is implemented, but the actual data yielding
        logic is delegated to another method.

        :return: Returns the instance itself as an iterator
        :rtype: object
        """
        return self

    def _filter_metadata(self, node: TextNode) -> TextNode:
        """
        Filters metadata of a given TextNode and its relationships to retain only
        specified metadata keys. The function removes metadata fields that are not
        in the allowed list of keys or metadata keys explicitly specified for the
        filtering process.

        The function operates on the `metadata` property of the node as well as on
        the metadata of any relationships associated with the node. This ensures
        that the metadata remains consistent with predefined filtering criteria.

        :param node: A TextNode object whose metadata and relationship metadata are
                     to be filtered.
        :type node: TextNode
        :return: The modified TextNode object with filtered metadata.
        :rtype: TextNode
        """

        def filter(metadata: Dict):
            """
            Handles operations related to file-based chunks for node handling, including
            filtering metadata based on specified conditions.

            Attributes:
                metadata_keys (Optional[List[str]]): A list of metadata keys to retain during
                    filtering. If None, all allowed keys are retained.

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
        Iterates over the files in the specified chunks directory and yields processed
        text nodes.

        This method reads files from the directory specified by `chunks_directory`
        and `collection_id`. Each file is processed to produce a `TextNode` object
        with metadata filtered using the `_filter_metadata` method.

        :yield: Processed `TextNode` objects created from the JSON content of each file.
        """
        directory_path = join(self.chunks_directory, self.collection_id)
        logger.debug(f'Reading chunks from directory: {directory_path}')
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    yield self._filter_metadata(TextNode.from_json(f.read()))

    def accept(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> Generator[BaseNode, None, None]:
        """
        Accepts a list of BaseNode instances and processes them based on their metadata, saving
        specific nodes to the chunks directory with metadata keys not containing `INDEX_KEY`.
        Yields each node regardless of its processing outcome.

        :param nodes: A list of BaseNode objects to process.
        :type nodes: List[BaseNode]
        :param kwargs: Additional keyword arguments that may be utilized during processing.
        :type kwargs: Any
        :return: A generator yielding each BaseNode after processing.
        :rtype: Generator[BaseNode, None, None]
        """
        for n in nodes:
            if not [key for key in [INDEX_KEY] if key in n.metadata]:
                chunk_output_path = join(
                    self.chunks_directory, self.collection_id, f'{n.node_id}.json'
                )
                logger.debug(f'Writing chunk to file: {chunk_output_path}')
                with open(chunk_output_path, 'w') as f:
                    json.dump(n.to_dict(), f, indent=4)
            yield n
