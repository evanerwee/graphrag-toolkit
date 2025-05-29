# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from datetime import datetime
from os.path import join
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY, TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY 

from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)

class FileBasedChunks(NodeHandler):
    """
    Handles the processing and management of text data chunks stored as files in a specified directory.

    The `FileBasedChunks` class provides functionality to handle and store chunks of text data, maintain
    metadata filtering, and process incoming `BaseNode` objects. Each chunk is represented as a file in
    the designated `chunks_directory`, and operations like filtering metadata, saving, and iterating
    through stored chunks are supported. The class can work with specific metadata keys and assigns a
    unique collection identifier for organizing files.

    Attributes:
        chunks_directory (str): Path to the directory where chunks are stored.
        collection_id (str): Unique identifier for the collection of chunks. Defaults to a timestamp if
            not provided.
        metadata_keys (Optional[List[str]]): List of metadata keys to retain during metadata filtering.
    """
    chunks_directory:str
    collection_id:str

    metadata_keys:Optional[List[str]]
    
    def __init__(self, 
                 chunks_directory:str, 
                 collection_id:Optional[str]=None,
                 metadata_keys:Optional[List[str]]=None):
        """
        Initializes the class with a directory for chunks, an optional collection ID, and
        a list of metadata keys. This constructor prepares the necessary configurations
        for managing data chunks, assigns a unique collection identifier if not provided,
        and handles directory preparation.

        Args:
            chunks_directory: The directory path where chunks will be stored.
            collection_id: Optional identifier for the collection. If not provided, a timestamp
                in the format '%Y%m%d-%H%M%S' will be generated and used as the collection ID.
            metadata_keys: Optional list of metadata keys for handling specific metadata related
                to the chunks.
        """
        super().__init__(
            chunks_directory=chunks_directory,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            metadata_keys=metadata_keys
        )
        self._prepare_directory()

    def _prepare_directory(self):
        """
        Creates a directory structure for storing collection-specific data if it does
        not already exist.

        This private method ensures that a directory path combining the base
        `chunks_directory` and the specific `collection_id` exists on the filesystem.
        If the directory does not exist, it is created. This setup is essential for
        managing and organizing files related to individual collections.

        Args:
            None

        Raises:
            None
        """
        directory_path = join(self.chunks_directory, self.collection_id)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def chunks(self):
        """
        Returns the current object, typically used as a placeholder or identity function.

        This method might be utilized in contexts where an object iterates over itself or where
        some form of chaining logic is required.

        Returns:
            object: The current instance of this object.
        """
        return self
    
    def _filter_metadata(self, node:TextNode) -> TextNode:
        """
        Filters the metadata of a TextNode instance to only retain specified keys or mandatory keys. Keys
        not listed in `self.metadata_keys` (if defined) or not among the mandatory keys will be deleted from
        the metadata of the node and its relationships.

        Args:
            node (TextNode): The node whose metadata needs to be filtered.

        Returns:
            TextNode: The node with its filtered metadata.
        """
        def filter(metadata:Dict):
            """
            Provides mechanisms to handle file-based chunks with the ability to filter their
            metadata based on specific keys defined within the context of the handler. This class
            processes nodes by manipulating the metadata to align with predefined conditions.
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
        Reads chunk files from a specified directory and yields processed data objects.

        Iterates over all files in a given directory and processes their contents. Each
        file's content is converted into a structured data object (`TextNode`) that is
        filtered to exclude metadata before being yielded. This is used to sequentially
        access and process a collection of data stored in chunk files.

        Returns:
            Generator: Yields structured data objects (`TextNode`) processed from file
            contents.

        Yields:
            TextNode: A filtered `TextNode` object created from reading and parsing the
            content of each chunk file.

        Attributes:
            chunks_directory (str): The base directory path containing chunked files.
            collection_id (str): The unique identifier for the specific collection of
                chunks being processed.

        Args:
            self: Instance of the object containing the iteration logic.
        """
        directory_path = join(self.chunks_directory, self.collection_id)
        logger.debug(f'Reading chunks from directory: {directory_path}')
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    yield self._filter_metadata(TextNode.from_json(f.read()))

    def accept(self, nodes: List[BaseNode], **kwargs: Any) -> Generator[BaseNode, None, None]:
        """Processes a list of nodes and writes certain nodes to disk while yielding them.

        This method iterates over a list of nodes. For each node, it checks if specific metadata
        keys are absent. If so, it writes the node's data to a JSON file within a predefined
        directory structure. Regardless of whether the node was written to disk, each node is
        yielded.

        Args:
            nodes (List[BaseNode]): A list of nodes to process.
            **kwargs (Any): Additional keyword arguments for potential use in processing.

        Yields:
            BaseNode: Each node from the input list, after optionally writing its data to disk.

        """
        for n in nodes:
            if not [key for key in [INDEX_KEY] if key in n.metadata]:
                chunk_output_path = join(self.chunks_directory, self.collection_id, f'{n.node_id}.json')
                logger.debug(f'Writing chunk to file: {chunk_output_path}')
                with open(chunk_output_path, 'w') as f:
                    json.dump(n.to_dict(), f, indent=4)
            yield n