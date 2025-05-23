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
    Facilitates operations related to handling S3-based documents by providing methods
    for processing, filtering metadata, iterating over documents, and storing them
    back to S3.

    This class is designed to serve as an interface for applications working with
    document nodes stored and managed in S3 buckets. It enables metadata filtering,
    standard processing workflows, and secure storage with optional encryption. The
    design supports both source document retrieval and the uploading of processed
    node data.

    :ivar region: The AWS region where the S3 bucket is deployed.
    :type region: str
    :ivar bucket_name: Name of the S3 bucket being accessed.
    :type bucket_name: str
    :ivar key_prefix: Path prefix in S3 bucket for storing or retrieving objects.
    :type key_prefix: str
    :ivar collection_id: Unique identifier for the collection managed in S3.
    :type collection_id: str
    :ivar s3_encryption_key_id: Optional encryption key ID for AWS KMS encryption of S3 objects.
    :type s3_encryption_key_id: Optional[str]
    :ivar metadata_keys: Optional list of metadata keys for filtering and managing
        associated data effectively.
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
        """Initializes an instance of the class with parameters for region,
        bucket name, key prefix, collection ID, S3 encryption key ID, and
        metadata keys. The collection ID defaults to the current timestamp if
        not provided.

        Args:
            region: The AWS region where the S3 bucket is hosted.
            bucket_name: The name of the S3 bucket to be used.
            key_prefix: The prefix to be applied to the keys for the objects stored.
            collection_id: The identifier for the collection. If not provided, defaults
                to a timestamp in '%Y%m%d-%H%M%S' format.
            s3_encryption_key_id: The ID of the encryption key used by S3 for encrypting
                objects.
            metadata_keys: A list of metadata keys associated with the collection. If
                None, metadata will not include additional keys.
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
        Represents the documentation for a generic function or method that
        returns its own instance. This function serves as a utility to return
        the object itself, which can be useful in chaining operations or
        maintaining fluency in method calls.

        :return: Returns the instance of the current object itself.
        :rtype: Self
        """
        return self

    def _filter_metadata(self, node: TextNode) -> TextNode:
        """
        Processes the provided TextNode object by filtering its metadata and associated
        relationship metadata based on predefined and optional criteria.

        This method mutates the metadata attached to the given TextNode by retaining only
        specific keys as defined in the filter logic. The filtering operation is applied
        to both the node's metadata and metadata of its relationships.

        :param node: The TextNode instance whose metadata is to be filtered.
        :type node: TextNode
        :return: The modified TextNode with filtered metadata.
        :rtype: TextNode
        """

        def filter(metadata: Dict):
            """
            Class responsible for handling operations involving S3 documents by filtering
            relevant metadata from input nodes. Designed to process and retain metadata
            that matches specific keys or pre-defined metadata configurations.

            :param metadata_keys: Optional list of keys that should be retained when
                processing metadata. If not provided, only keys predefined in the
                PROPOSITIONS_KEY, TOPICS_KEY, and INDEX_KEY constants are retained.

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
        Iterates over source documents stored in an S3 bucket and yields them after
        processing. This method retrieves and processes document data by paginating
        through S3 objects and filtering their metadata.

        The documents are represented by nodes, which are extracted from S3 path
        prefixes, downloaded as chunks, and parsed from JSON format before filtering.
        Each document is then yielded as a `SourceDocument` containing the nodes.

        :param self:
            The instance of the class containing S3 configurations and context-specific
            attributes such as `bucket_name`, `collection_id`, and `key_prefix`.

        :returns:
            A generator that yields `SourceDocument` objects, where each source document
            comprises a list of nodes parsed and filtered from the S3 data.

        :raises:
            This method does not explicitly describe raised exceptions, but S3 client
            operations, I/O operations, or JSON parsing may raise exceptions during
            execution.
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
        __call__ method processes a list of source types and utilizes the provided
        arguments to transform them into a list of source documents. This
        method acts as a wrapper that combines source type conversion and
        processing logic to generate source documents effectively.

        :param nodes: List of source types to be processed.
        :type nodes: List[SourceType]
        :param kwargs: Additional arguments that influence the processing of
                       source types.
        :type kwargs: Any
        :return: A list of processed source documents derived from the
                 given source types.
        :rtype: List[SourceDocument]
        """
        return [
            n for n in self.accept(source_documents_from_source_types(nodes), **kwargs)
        ]

    def accept(
        self, source_documents: List[SourceDocument], **kwargs: Any
    ) -> Generator[SourceDocument, None, None]:
        """
        Processes and uploads a list of source documents and their nodes to an S3 bucket.

        The method iterates over the provided source documents and writes each document's
        nodes to an S3 bucket as JSON files. The key prefix structure and additional details
        required for S3 uploading are configured using the class attributes such as
        `bucket_name`, `key_prefix`, `collection_id`, and `s3_encryption_key_id`. The
        method also leverages AWS KMS encryption for secure storage of the objects
        where applicable.

        :param source_documents: List of SourceDocument objects to be processed and stored.
        :type source_documents: List[SourceDocument]
        :param kwargs: Optional additional parameters.
        :type kwargs: Any
        :return: A generator to yield each processed SourceDocument after storage.
        :rtype: Generator[SourceDocument, None, None]
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
