# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging

from os.path import join
from datetime import datetime
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
from graphrag_toolkit.lexical_graph import GraphRAGConfig

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class S3BasedDocs(NodeHandler):
    """
    Class responsible for handling operations involving S3 documents. This class provides
    functionality for managing, processing, and storing data in an AWS S3 bucket. It allows
    filtering of metadata within nodes, retrieval of source documents, iteration over S3
    data, and handling of S3 encryption for added security.

    The class offers methods that assist in retrieving data from S3, filtering it, and
    writing processed information back to the bucket in a structured and secure manner.
    It integrates seamlessly with the S3 API for object listing, downloading, and uploading.

    :ivar region: The AWS region where the S3 bucket is located.
    :type region: str
    :ivar bucket_name: The name of the S3 bucket used for storing documents.
    :type bucket_name: str
    :ivar key_prefix: The prefix for keys in the S3 bucket, defining the folder structure.
    :type key_prefix: str
    :ivar collection_id: The unique identifier for the collection being managed in S3.
    :type collection_id: str
    :ivar s3_encryption_key_id: The optional KMS encryption key ID for securing S3 objects.
    :type s3_encryption_key_id: Optional[str]
    :ivar metadata_keys: The optional list of metadata keys to filter when processing data.
    :type metadata_keys: Optional[List[str]]
    """

    region: str
    bucket_name: str
    key_prefix: str
    collection_id: str
    s3_encryption_key_id: Optional[str] = None
    metadata_keys: Optional[List[str]] = None

    def __init__(
        self,
        region: str,
        bucket_name: str,
        key_prefix: str,
        collection_id: Optional[str] = None,
        s3_encryption_key_id: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
    ):
        """
        Initializes the instance with the specified region, bucket name, key
        prefix, and optional parameters for collection ID, S3 encryption key ID,
        and metadata keys. The collection ID defaults to the current timestamp
        formatted as '%Y%m%d-%H%M%S' if not provided.

        :param region: AWS region where the S3 bucket is located.
        :param bucket_name: Name of the S3 bucket to be used.
        :param key_prefix: Prefix for the S3 keys.
        :param collection_id: Unique identifier for the collection. Defaults
            to the current timestamp in the '%Y%m%d-%H%M%S' format if not provided.
        :param s3_encryption_key_id: Optional encryption key ID for securing S3 keys
            in cases where server-side encryption is required.
        :param metadata_keys: Optional list of metadata keys to associate with the
            S3 objects.
        :type region: str
        :type bucket_name: str
        :type key_prefix: str
        :type collection_id: Optional[str]
        :type s3_encryption_key_id: Optional[str]
        :type metadata_keys: Optional[List[str]]
        """
        super().__init__(
            region=region,
            bucket_name=bucket_name,
            key_prefix=key_prefix,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            s3_encryption_key_id=s3_encryption_key_id,
            metadata_keys=metadata_keys,
        )

    def docs(self):
        """
        This method returns the instance of the object itself. It is typically
        used for cases where chaining or fluent interfaces are desired. No operations
        are performed in this method beyond returning `self`.

        :return: The instance of the object itself.
        :rtype: The same object instance

        """
        return self

    def _filter_metadata(self, node: TextNode) -> TextNode:
        """
        Filters metadata from a TextNode and its associated relationships.
        The function modifies the `metadata` attributes of the input `node`
        and any nested relationships to retain specific keys, either those
        defined in the constants PROPOSITIONS_KEY, TOPICS_KEY, and INDEX_KEY,
        or those specified explicitly in `self.metadata_keys`.

        :param node: The input TextNode object whose metadata and relationships'
            metadata need to be filtered.
        :type node: TextNode
        :return: The updated TextNode with filtered metadata.
        :rtype: TextNode
        """

        def filter(metadata: Dict):
            """
            Handles S3-based document processing utilizing metadata filtering mechanisms.
            This class is an implementation of NodeHandler specialized for specific metadata
            management. It includes functionality to process, handle, and filter metadata
            based on predefined criteria.

            Attributes:
                metadata_keys: Optional list of metadata keys to retain during filtering.

            :param metadata: Dictionary containing key-value pairs of document metadata.
            :type metadata: Dict
            :return: Filtered node with the processed metadata.
            :rtype: TextNode
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
        Provides an iterator to process and yield source documents, stored in an S3
        bucket, where each source document consists of multiple nodes downloaded
        and processed in chunks.

        This method fetches a collection of source documents based on a specified
        S3 bucket, key prefix, and collection ID. Each source document prefix is
        iterated upon to retrieve its associated content in chunks. The chunks
        are downloaded, decoded, and processed to construct text nodes, which are
        further encapsulated into SourceDocument objects for yielding. Logging is
        used to provide debugging information about the ongoing operations.

        :raises KeyError: If expected keys are not found in the response from S3.
        :raises botocore.exceptions.BotoCoreError: For errors raised by Boto3
            or botocore in accessing S3.
        :raises IOError: For I/O errors during file download or processing.

        :yield: SourceDocument instances containing nodes retrieved and processed
            from the S3 storage.
        """
        s3_client = GraphRAGConfig.s3

        collection_path = join(self.key_prefix, self.collection_id, '')

        logger.debug(
            f'Getting source documents from S3: [bucket: {self.bucket_name}, key: {collection_path}]'
        )

        paginator = s3_client.get_paginator('list_objects_v2')
        source_doc_pages = paginator.paginate(
            Bucket=self.bucket_name, Prefix=collection_path, Delimiter='/'
        )

        source_doc_prefixes = [
            source_doc_obj['Prefix']
            for source_doc_page in source_doc_pages
            for source_doc_obj in source_doc_page['CommonPrefixes']
        ]

        for source_doc_prefix in source_doc_prefixes:

            nodes = []

            chunk_pages = paginator.paginate(
                Bucket=self.bucket_name, Prefix=source_doc_prefix
            )

            chunk_keys = [
                chunk_obj['Key']
                for chunk_page in chunk_pages
                for chunk_obj in chunk_page['Contents']
            ]

            for chunk_key in chunk_keys:
                with io.BytesIO() as io_stream:
                    s3_client.download_fileobj(self.bucket_name, chunk_key, io_stream)
                    io_stream.seek(0)
                    data = io_stream.read().decode('UTF-8')
                    nodes.append(self._filter_metadata(TextNode.from_json(data)))

            logger.debug(
                f'Yielding source document [source: {source_doc_prefix}, num_nodes: {len(nodes)}]'
            )

            yield SourceDocument(nodes=nodes)

    def __call__(self, nodes: List[SourceType], **kwargs: Any) -> List[SourceDocument]:
        """
        Processes a list of source types and returns a list of source documents. It first
        transforms the source types into source documents, and then applies any additional
        processing specified by subclass implementations through the `accept` method.

        :param nodes: A list of source types to be processed.
        :type nodes: List[SourceType]
        :param kwargs: Additional keyword arguments passed to the `accept` method.
        :type kwargs: Any
        :return: A list of processed source documents.
        :rtype: List[SourceDocument]
        """
        return [
            n for n in self.accept(source_documents_from_source_types(nodes), **kwargs)
        ]

    def accept(
        self, source_documents: List[SourceDocument], **kwargs: Any
    ) -> Generator[SourceDocument, None, None]:
        """
        Processes a list of source documents and writes their metadata chunks to an S3 bucket.

        Iterates through the provided source documents and generates an output stream
        using a generator pattern. Each document and its nodes are processed and stored
        in an S3 bucket with proper encryption settings if configured. Supports both KMS
        and AES256 encryption for added security. Once a source document's nodes are
        completely written to S3, the method yields the processed document.

        :param source_documents:
            A list of source documents to process and upload to an S3 bucket.
            Each document contains metadata that is iterated for storage.
        :param kwargs:
            Additional key-value arguments that can be passed for extended or
            method-specific configuration, if applicable.

        :return:
            A generator that yields processed `SourceDocument` objects after their
            respective metadata chunks are successfully written to S3 for storage.
        """

        s3_client = GraphRAGConfig.s3

        for source_document in source_documents:

            root_path = join(
                self.key_prefix, self.collection_id, source_document.source_id()
            )
            logger.debug(
                f'Writing source document to S3 [bucket: {self.bucket_name}, prefix: {root_path}]'
            )

            for n in source_document.nodes:
                if not [key for key in [INDEX_KEY] if key in n.metadata]:

                    chunk_output_path = join(root_path, f'{n.node_id}.json')

                    logger.debug(
                        f'Writing chunk to S3: [bucket: {self.bucket_name}, key: {chunk_output_path}]'
                    )

                    if self.s3_encryption_key_id:
                        s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=chunk_output_path,
                            Body=(
                                bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))
                            ),
                            ContentType='application/json',
                            ServerSideEncryption='aws:kms',
                            SSEKMSKeyId=self.s3_encryption_key_id,
                        )
                    else:
                        s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=chunk_output_path,
                            Body=(
                                bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))
                            ),
                            ContentType='application/json',
                            ServerSideEncryption='AES256',
                        )

            yield source_document
