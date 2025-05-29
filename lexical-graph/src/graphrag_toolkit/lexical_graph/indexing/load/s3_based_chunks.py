# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging

from os.path import join
from datetime import datetime
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY, TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY
from graphrag_toolkit.lexical_graph import GraphRAGConfig


from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)

class S3BasedChunks(NodeHandler):
    """
    Handles chunk management with Amazon S3 as the backend.

    The S3BasedChunks class facilitates the storage, retrieval, and management
    of chunks in an Amazon S3 bucket. It allows for filtering metadata, iterating
    over stored chunks in the S3 bucket, and writing new chunks to the bucket.
    This class leverages the S3 client from GraphRAGConfig for handling S3 operations
    and supports optional server-side encryption configurations.

    Attributes:
        region (str): AWS region where the S3 bucket is located.
        bucket_name (str): Name of the S3 bucket used for chunk storage.
        key_prefix (str): Prefix for keys in S3 to logically organize chunks.
        collection_id (str): Identifier for the collection of chunks, defaults to
            the current timestamp if not provided.
        s3_encryption_key_id (Optional[str]): AWS KMS key ID for server-side encryption,
            if specified.
        metadata_keys (Optional[List[str]]): List of metadata keys to retain
            during metadata filtering. If None, no keys are specifically retained.
    """
    region:str
    bucket_name:str
    key_prefix:str
    collection_id:str
    s3_encryption_key_id:Optional[str]=None
    metadata_keys:Optional[List[str]]=None

    def __init__(self, 
                 region:str, 
                 bucket_name:str, 
                 key_prefix:str, 
                 collection_id:Optional[str]=None,
                 s3_encryption_key_id:Optional[str]=None, 
                 metadata_keys:Optional[List[str]]=None):
        """
        Initializes the object with specified configuration parameters allowing for
        interaction with S3 and enabling metadata customization. This setup also
        ensures encryption options and collection-specific handling.

        Args:
            region: The AWS region associated with the S3 service.
            bucket_name: The name of the S3 bucket to use for storage.
            key_prefix: The prefix (folder path) for keys in the S3 bucket.
            collection_id: Optional unique identifier for the collection. Defaults
                to the current timestamp if not specified.
            s3_encryption_key_id: Optional ID of the encryption key to use for S3 object
                encryption.
            metadata_keys: Optional list of keys indicating which metadata are
                utilized for objects in the collection.

        """
        super().__init__(
            region=region,
            bucket_name=bucket_name,
            key_prefix=key_prefix,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            s3_encryption_key_id=s3_encryption_key_id,
            metadata_keys=metadata_keys
        )

    def chunks(self):
        """
        Returns the object itself as an iterable object.

        This method provides the ability to iterate directly over the object rather than
        returning a separate iterator or collection. It is useful when implementing a
        custom iterable class where the object itself can serve as its own iterable.

        Returns:
            self: The object itself, allowing iteration over its content.
        """
        return self
    
    def _filter_metadata(self, node:TextNode) -> TextNode:
        """
        Filters metadata from a given TextNode, retaining only specified keys if applicable.

        The function removes metadata entries from a TextNode that do not match a predefined
        set of keys. If `self.metadata_keys` is defined, only metadata keys within that list
        are retained. The filtering operation applies recursively to both the node's metadata
        and the metadata of its relationships.

        Args:
            node (TextNode): The node whose metadata is to be filtered.

        Returns:
            TextNode: The node with metadata filtered according to the criteria.
        """
        def filter(metadata:Dict):
            """
            Handles the processing of nodes with metadata stored in an S3-based system.
            Filters metadata of a given node based on predefined criteria, removing
            unnecessary or irrelevant keys.

            Attributes:
                metadata_keys (Optional[List[str]]): Specifies a list of metadata keys to
                    retain during the filtering process. If None, all metadata keys
                    allowed by the filtering logic will be retained.
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
        Overrides the `__iter__` method to yield processed data retrieved from an S3 bucket. The
        method fetches objects from a specific S3 bucket and prefix, downloads their content,
        deserializes the data, and applies a metadata filtering process before yielding the results.

        Args:
            None

        Yields:
            TextNode: The deserialized and processed node data, filtered based on specific
            metadata conditions.

        Raises:
            Any exceptions related to S3 operations or data processing.

        """
        s3_client = GraphRAGConfig.s3  # Uses dynamic __getattr__

        collection_path = join(self.key_prefix, self.collection_id)

        logger.debug(f'Getting chunks from S3: [bucket: {self.bucket_name}, key: {collection_path}]')

        chunks = s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=collection_path)

        for obj in chunks.get('Contents', []):
            key = obj['Key']
            
            if key.endswith('/'):
                continue

            with io.BytesIO() as io_stream:
                s3_client.download_fileobj(self.bucket_name, key, io_stream)        
                io_stream.seek(0)
                data = io_stream.read().decode('UTF-8')
                yield self._filter_metadata(TextNode.from_json(data))

    def accept(self, nodes: List[BaseNode], **kwargs: Any) -> Generator[BaseNode, None, None]:
        """
        Processes and uploads nodes to an S3 bucket, while also yielding each individual
        node after processing. Utilizes server-side encryption for data storage on S3
        and supports both KMS managed keys and AES256 encryption.

        Args:
            nodes (List[BaseNode]): A list of `BaseNode` objects to be processed and
                uploaded to the S3 bucket.
            **kwargs (Any): Additional keyword arguments that might be passed to this
                function for extended functionality.

        Yields:
            BaseNode: Yields each `BaseNode` object from the provided `nodes` list after
                processing.

        """
        s3_client = GraphRAGConfig.s3
        for n in nodes:
            if not [key for key in [INDEX_KEY] if key in n.metadata]:

                chunk_output_path = join(self.key_prefix,  self.collection_id, f'{n.node_id}.json')
                
                logger.debug(f'Writing chunk to S3: [bucket: {self.bucket_name}, key: {chunk_output_path}]')

                if self.s3_encryption_key_id:
                    s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=chunk_output_path,
                        Body=(bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))),
                        ContentType='application/json',
                        ServerSideEncryption='aws:kms',
                        SSEKMSKeyId=self.s3_encryption_key_id
                    )
                else:
                    s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=chunk_output_path,
                        Body=(bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))),
                        ContentType='application/json',
                        ServerSideEncryption='AES256'
                    )

            yield n