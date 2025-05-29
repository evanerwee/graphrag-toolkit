# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging

from os.path import join
from datetime import datetime
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.constants import (
    PROPOSITIONS_KEY,
    TOPICS_KEY,
)
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY
from graphrag_toolkit.lexical_graph import GraphRAGConfig


from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)


class S3BasedChunks(NodeHandler):
    """
    Manages the interaction with an S3-based storage system to handle chunks of
    data, providing functionality for serialization, metadata filtering,
    iteration, and upload operations.

    This class is intended for managing data chunks in an S3 storage environment
    through a defined bucket, key prefix, and optional collection ID. It allows
    for filtering metadata, processing nodes, server-side encryption during data
    upload, and custom iteration functionality.

    :ivar region: Represents the AWS region associated with the S3 service.
    :type region: str
    :ivar bucket_name: The name of the S3 bucket being used for data storage.
    :type bucket_name: str
    :ivar key_prefix: The prefix or folder structure for grouping items in S3.
    :type key_prefix: str
    :ivar collection_id: A unique identifier for referring to a specific data
        collection.
    :type collection_id: str
    :ivar s3_encryption_key_id: Optional KMS encryption key ID for securing S3
        objects.
    :type s3_encryption_key_id: Optional[str]
    :ivar metadata_keys: Optional list defining metadata keys to retain for
        filtering.
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
        Initializes an instance with the given parameters for file storage or data
        collection configuration. This initialization sets up region-specific S3
        bucket information, encryption, data organization prefixes, and metadata
        keys for efficient management.

        :param region: The AWS region where the S3 bucket is located.
        :type region: str
        :param bucket_name: The name of the S3 bucket being accessed.
        :type bucket_name: str
        :param key_prefix: The prefix to be used for objects stored in the S3 bucket.
        :type key_prefix: str
        :param collection_id: The identifier for a specific data collection. Defaults
            to a timestamp (`'%Y%m%d-%H%M%S'`) if not provided.
        :type collection_id: Optional[str]
        :param s3_encryption_key_id: The ID of the key used for encryption in S3.
            This is optional and allows for securing stored objects.
        :type s3_encryption_key_id: Optional[str]
        :param metadata_keys: A list of metadata keys that are relevant for the
            storage process or data organization.
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

    def chunks(self):
        """
        Provides an iterable that returns itself.

        This function is designed to return the instance it is called from, enabling
        it to be traversed as an iterable. It is a simple way to create an object that
        is compatible with Python's iteration protocols without any additional
        complexity.

        :return: The instance of the class to allow iteration.
        :rtype: object
        """
        return self

    def _filter_metadata(self, node: TextNode) -> TextNode:
        """
        Filters metadata of a TextNode and its relationships based on predefined criteria.
        The filtering process removes unnecessary or irrelevant metadata keys, ensuring that
        only useful keys are retained. This function supports an S3-based system for
        metadata storage.

        :param node: The TextNode instance whose metadata and associated relationship
            metadata need to be filtered. It contains a dictionary of metadata and
            optionally relationships with their own metadata.
        :type node: TextNode
        :return: The TextNode instance with filtered metadata and relationship metadata.
        :rtype: TextNode
        """

        def filter(metadata: Dict):
            """
            Handles processing of text nodes based on metadata filtering.

            This class is designed to manage and process nodes using specific metadata
            filtering criteria. It inherits from `NodeHandler`. Extensive operations
            or configurations should be defined externally.

            Attributes:
                metadata_keys (Optional[List[str]]): List of metadata keys to retain during
                    filtering. Any metadata keys not present in this list will be removed,
                    unless they are keys specifically allowed internally.
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
        Iterates over data chunks retrieved from an S3 bucket, filters metadata,
        and yields deserialized TextNode objects.

        The method accesses the specified S3 bucket and the collection path to
        retrieve objects stored as JSON. Each retrieved object is downloaded,
        decoded, and converted to a TextNode object. Metadata filtering is
        applied before yielding the resulting TextNode.

        :yield: Filtered TextNode objects obtained from the S3 bucket.

        :raises KeyError: If the required 'Contents' key is missing in the
            response obtained from the S3 client while listing objects.
        :raises ClientError: If an error occurs during the interaction with
            the S3 client while downloading objects.
        """
        s3_client = GraphRAGConfig.s3  # Uses dynamic __getattr__

        collection_path = join(self.key_prefix, self.collection_id)

        logger.debug(
            f'Getting chunks from S3: [bucket: {self.bucket_name}, key: {collection_path}]'
        )

        chunks = s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=collection_path
        )

        for obj in chunks.get('Contents', []):
            key = obj['Key']

            if key.endswith('/'):
                continue

            with io.BytesIO() as io_stream:
                s3_client.download_fileobj(self.bucket_name, key, io_stream)
                io_stream.seek(0)
                data = io_stream.read().decode('UTF-8')
                yield self._filter_metadata(TextNode.from_json(data))

    def accept(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> Generator[BaseNode, None, None]:
        """
        Generates nodes from the input list, processes them, and writes chunks
        containing their metadata to Amazon S3 based on the provided configuration.

        This function iterates over the given nodes, examining their metadata to
        check for a required key. If the key does not exist in the node metadata, the
        node is serialized in JSON format and uploaded to the configured S3 bucket.
        The node's corresponding metadata chunk is stored using either AWS KMS
        encryption or AES256 depending on the given configuration. Finally, it
        yields each node after processing.

        :param nodes: A list of nodes to process and potentially upload to S3. Each
            node in the list should conform to the expected structure and type
            defined by BaseNode.
        :param kwargs: Additional keyword arguments that may be used for future
            extension or additional processing in this function.
        :return: A generator that yields each processed node from the input list
            sequentially, after its storage is handled (if applicable).
        """
        s3_client = GraphRAGConfig.s3
        for n in nodes:
            if not [key for key in [INDEX_KEY] if key in n.metadata]:

                chunk_output_path = join(
                    self.key_prefix, self.collection_id, f'{n.node_id}.json'
                )

                logger.debug(
                    f'Writing chunk to S3: [bucket: {self.bucket_name}, key: {chunk_output_path}]'
                )

                if self.s3_encryption_key_id:
                    s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=chunk_output_path,
                        Body=(bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))),
                        ContentType='application/json',
                        ServerSideEncryption='aws:kms',
                        SSEKMSKeyId=self.s3_encryption_key_id,
                    )
                else:
                    s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=chunk_output_path,
                        Body=(bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))),
                        ContentType='application/json',
                        ServerSideEncryption='AES256',
                    )

            yield n
