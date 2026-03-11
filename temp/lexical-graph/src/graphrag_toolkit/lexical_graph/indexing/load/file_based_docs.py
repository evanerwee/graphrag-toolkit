# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from datetime import datetime
from os.path import join
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument, SourceType, source_documents_from_source_types
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY, TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY 

from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)

class FileBasedDocs(NodeHandler):
    """Handler for file-based document processing.

    This class is designed to process and manage source documents stored within a
    file system. It enables the preparation, reading, filtering, and writing of
    documents to and from specified directories. It facilitates document handling
    by utilizing `TextNode` and `SourceDocument` structures, while providing
    customizable metadata filtering for nodes and their relationships.

    Attributes:
        docs_directory (str): The directory path where documents are stored.
        collection_id (str): The identifier of the document collection. If not
            provided, a timestamp-based ID is generated.
        metadata_keys (Optional[List[str]]): A list of allowed metadata keys. Only
            these keys will be retained during metadata filtering.
    """
    docs_directory:str
    collection_id:str

    metadata_keys:Optional[List[str]]
    
    def __init__(self, 
                 docs_directory:str, 
                 collection_id:Optional[str]=None,
                 metadata_keys:Optional[List[str]]=None):
        """
        Initializes the object with the specified documents directory, collection ID, and
        optional metadata keys. It also prepares the directory for the given collection ID
        within the documents directory.

        Args:
            docs_directory (str): The path to the directory where the documents will be stored.
            collection_id (Optional[str]): The ID of the collection. If not provided, defaults
                to a timestamp in the format 'YYYYMMDD-HHMMSS'.
            metadata_keys (Optional[List[str]]): A list of metadata keys to associate with
                the documents in the collection.
        """
        super().__init__(
            docs_directory=docs_directory,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            metadata_keys=metadata_keys
        )
        self._prepare_directory(join(self.docs_directory, self.collection_id))

    def _prepare_directory(self, directory_path):
        """
        Creates a directory if it does not already exist.

        This method checks if the directory at the specified path exists, and if
        not, it creates the directory along with any necessary intermediate-level
        directories. It ensures that the provided path is ready for use without
        requiring prior manual setup.

        Args:
            directory_path (str): The file path of the directory to be verified
                or created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def docs(self):
        """
        A method that provides documentation in the form of a returned value.

        This method is designed to demonstrate a specific behavior of returning
        the object itself. It does not perform additional computations or
        operations. Typically used in cases requiring fluent interfaces or
        method chaining.

        Returns:
            object: The instance of the object itself.
        """
        return self
    
    def _filter_metadata(self, node:TextNode) -> TextNode:
        """Filters specific metadata from the given TextNode based on allowed keys.

        This method modifies the `metadata` of the input `TextNode` and its
        relationships by removing any keys that are not in the allowed list
        [PROPOSITIONS_KEY, TOPICS_KEY, INDEX_KEY] or those not present in
        `self.metadata_keys` (if defined). It retains only the permitted metadata
        keys within the node and its relationships.

        Args:
            node (TextNode): The TextNode whose metadata is filtered.

        Returns:
            TextNode: The modified TextNode with filtered metadata.
        """
        def filter(metadata:Dict):
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
        Iterates through the directories and files to yield SourceDocument objects populated
        with nodes created from the file contents.

        This method traverses the structure within a given directory path corresponding
        to the collection ID, processing files located within subdirectories to extract
        data and generate SourceDocument objects.

        Yields:
            SourceDocument: An object containing nodes created from the JSON file contents
            found in the directory structure.

        Args:
            None
        """
        directory_path = join(self.docs_directory, self.collection_id)
        
        logger.debug(f'Reading source documents from directory: {directory_path}')
        
        source_document_directory_paths = [f.path for f in os.scandir(directory_path) if f.is_dir()]
        
        for source_document_directory_path in source_document_directory_paths:
            nodes = []
            for filename in os.listdir(source_document_directory_path):
                file_path = os.path.join(source_document_directory_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path) as f:
                        nodes.append(self._filter_metadata(TextNode.from_json(f.read())))
            yield SourceDocument(nodes=nodes)

    def __call__(self, nodes: List[SourceType], **kwargs: Any) -> List[SourceDocument]:
        return [n for n in self.accept(source_documents_from_source_types(nodes), **kwargs)]

    def accept(self, source_documents: List[SourceDocument], **kwargs: Any) -> Generator[SourceDocument, None, None]:
        """
        This method processes a list of source documents, organizes them into directories based
        on their source ID, and writes individual nodes of each document to separate JSON files.
        It then yields the processed source documents.

        Args:
            source_documents (List[SourceDocument]): A list of source documents to be processed.
            **kwargs (Any): Arbitrary keyword arguments that might be used when processing
                the source documents.

        Yields:
            SourceDocument: The processed source document after its nodes have been written
                to corresponding JSON files in the directory structure.
        """
        for source_document in source_documents:
            directory_path =  join(self.docs_directory, self.collection_id, source_document.source_id())
            self._prepare_directory(directory_path)
            logger.debug(f'Writing source document to directory: {directory_path}')
            for node in source_document.nodes:
                if not [key for key in [INDEX_KEY] if key in node.metadata]:
                    chunk_output_path = join(directory_path, f'{node.node_id}.json')
                    logger.debug(f'Writing chunk to file: {chunk_output_path}')
                    with open(chunk_output_path, 'w') as f:
                        json.dump(node.to_dict(), f, indent=4)
            yield source_document
