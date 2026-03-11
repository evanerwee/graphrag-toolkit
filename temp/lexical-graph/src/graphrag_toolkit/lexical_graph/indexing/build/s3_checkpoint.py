# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3-based checkpoint storage for distributed and resumable workflows.

This module provides S3-backed checkpoint functionality that enables:
- Distributed workflows across multiple workers
- Persistent checkpoints that survive container restarts
- Cross-region checkpoint sharing for disaster recovery

Usage:
    from graphrag_toolkit.lexical_graph.indexing.build import S3Checkpoint
    
    checkpoint = S3Checkpoint(
        checkpoint_name="build-checkpoint",
        bucket="my-output-bucket",
        prefix="checkpoints",
        region="us-east-1"
    )
    
    # Use with LexicalGraphIndex
    graph_index.build(docs, checkpoint=checkpoint)
"""

import logging
from typing import Any, List, Optional

from botocore.exceptions import ClientError

from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.tenant_id import TenantId
from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY
from graphrag_toolkit.lexical_graph.indexing.build.checkpoint import (
    DoNotCheckpoint,
    SAVEPOINT_ROOT_DIR,
)

from llama_index.core.schema import TransformComponent, BaseNode

logger = logging.getLogger(__name__)


class S3CheckpointFilter(TransformComponent, DoNotCheckpoint):
    """S3-based checkpoint filter for distributed workflows.
    
    Filters nodes based on checkpoint existence in S3, enabling resumable
    operations across distributed workers.
    
    Uses GraphRAGConfig.s3 for S3 operations, which is pickle-safe since
    the client is accessed from the singleton rather than stored.
    """
    
    checkpoint_name: str
    bucket: str
    prefix: str
    inner: TransformComponent
    tenant_id: TenantId
    
    def __init__(
        self,
        checkpoint_name: str,
        bucket: str,
        prefix: str,
        inner: TransformComponent,
        tenant_id: TenantId,
    ):
        super().__init__(
            checkpoint_name=checkpoint_name,
            bucket=bucket,
            prefix=prefix,
            inner=inner,
            tenant_id=tenant_id,
        )

    def _get_checkpoint_key(self, node_id: str) -> str:
        """Build the S3 key for a checkpoint."""
        return f"{self.prefix}/{SAVEPOINT_ROOT_DIR}/{self.checkpoint_name}/{node_id}"
    
    def checkpoint_does_not_exist(self, node_id: str) -> bool:
        """Check if checkpoint exists in S3 for the given node.
        
        Args:
            node_id: Identifier of the node to check.
            
        Returns:
            True if checkpoint does NOT exist (node should be processed),
            False if checkpoint exists (node should be skipped).
        """
        tenant_node_id = self.tenant_id.rewrite_id(node_id)
        checkpoint_key = self._get_checkpoint_key(tenant_node_id)
        
        try:
            GraphRAGConfig.s3.head_object(Bucket=self.bucket, Key=checkpoint_key)
            logger.debug(
                f'Ignoring node because S3 checkpoint exists '
                f'[node_id: {tenant_node_id}, checkpoint: {self.checkpoint_name}, '
                f's3://{self.bucket}/{checkpoint_key}, component: {type(self.inner).__name__}]'
            )
            return False
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.debug(
                    f'Including node [node_id: {tenant_node_id}, '
                    f'checkpoint: {self.checkpoint_name}, component: {type(self.inner).__name__}]'
                )
                return True
            else:
                logger.warning(f'S3 error checking checkpoint: {e}')
                # On error, include the node to be safe
                return True
    
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Filter nodes based on S3 checkpoint existence."""
        discarded_count = 0
        filtered_nodes = []
        
        for node in nodes:
            if self.checkpoint_does_not_exist(node.id_):
                filtered_nodes.append(node)
            else:
                discarded_count += 1
        
        if discarded_count > 0:
            logger.info(
                f'[{type(self.inner).__name__}] Discarded {discarded_count} out of '
                f'{discarded_count + len(filtered_nodes)} nodes because they have '
                f'already been checkpointed in S3'
            )
        
        return self.inner.__call__(filtered_nodes, **kwargs)


class S3CheckpointWriter(NodeHandler):
    """S3-based checkpoint writer for distributed workflows.
    
    Writes checkpoint markers to S3 after successful node processing.
    
    Uses GraphRAGConfig.s3 for S3 operations, which is pickle-safe since
    the client is accessed from the singleton rather than stored.
    """
    
    checkpoint_name: str
    bucket: str
    prefix: str
    inner: NodeHandler
    
    def __init__(
        self,
        checkpoint_name: str,
        bucket: str,
        prefix: str,
        inner: NodeHandler,
    ):
        super().__init__(
            checkpoint_name=checkpoint_name,
            bucket=bucket,
            prefix=prefix,
            inner=inner,
        )
    
    def _get_checkpoint_key(self, node_id: str) -> str:
        """Build the S3 key for a checkpoint."""
        return f"{self.prefix}/{SAVEPOINT_ROOT_DIR}/{self.checkpoint_name}/{node_id}"
    
    def _write_checkpoint(self, node_id: str) -> None:
        """Write a checkpoint marker to S3."""
        checkpoint_key = self._get_checkpoint_key(node_id)
        try:
            GraphRAGConfig.s3.put_object(
                Bucket=self.bucket,
                Key=checkpoint_key,
                Body=b'',  # Empty marker file
                ContentType='application/octet-stream',
            )
            logger.debug(f'Wrote S3 checkpoint: s3://{self.bucket}/{checkpoint_key}')
        except ClientError as e:
            logger.error(f'Failed to write S3 checkpoint: {e}')
            raise
    
    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        """Process nodes and write S3 checkpoints for successful ones."""
        for node in self.inner.accept(nodes, **kwargs):
            node_id = node.node_id
            if [key for key in [INDEX_KEY] if key in node.metadata]:
                logger.debug(
                    f'Non-checkpointable node [checkpoint: {self.checkpoint_name}, '
                    f'node_id: {node_id}, component: {type(self.inner).__name__}]'
                )
            else:
                logger.debug(
                    f'Checkpointable node [checkpoint: {self.checkpoint_name}, '
                    f'node_id: {node_id}, component: {type(self.inner).__name__}]'
                )
                self._write_checkpoint(node_id)
            yield node


class S3Checkpoint:
    """S3-based checkpoint manager for distributed and resumable workflows.
    
    This class provides the same interface as the local Checkpoint class but
    stores checkpoint markers in S3, enabling:
    - Distributed workflows across multiple workers/containers
    - Persistent checkpoints that survive container restarts
    - Cross-region checkpoint sharing
    
    Attributes:
        checkpoint_name: Name of the checkpoint (used in S3 key path).
        bucket: S3 bucket name for checkpoint storage.
        prefix: S3 key prefix for organizing checkpoints.
        enabled: Whether checkpointing is enabled.
        
    Example:
        checkpoint = S3Checkpoint(
            checkpoint_name="build-checkpoint",
            bucket="my-workflow-bucket",
            prefix="checkpoints/tenant-123",
        )
        
        graph_index.build(docs, checkpoint=checkpoint)
    """
    
    def __init__(
        self,
        checkpoint_name: str,
        bucket: str,
        prefix: str = "checkpoints",
        enabled: bool = True,
    ):
        """Initialize S3Checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint (e.g., "build-checkpoint").
            bucket: S3 bucket name for storing checkpoints.
            prefix: S3 key prefix for organizing checkpoints. Defaults to "checkpoints".
            enabled: Whether checkpointing is enabled. Defaults to True.
        """
        self.checkpoint_name = checkpoint_name
        self.bucket = bucket
        self.prefix = prefix
        self.enabled = enabled
        
        # Log initialization
        checkpoint_path = f"s3://{bucket}/{prefix}/{SAVEPOINT_ROOT_DIR}/{checkpoint_name}/"
        logger.info(f'Initialized S3Checkpoint [checkpoint: {checkpoint_name}, path: {checkpoint_path}]')
    
    def add_filter(self, o: Any, tenant_id: TenantId) -> Any:
        """Add S3 checkpoint filter to a transform component.
        
        Wraps the provided transform component with an S3-based checkpoint filter
        if conditions are met (enabled, is TransformComponent, not DoNotCheckpoint).
        
        Args:
            o: The transform component to potentially wrap.
            tenant_id: Tenant ID for multi-tenant isolation.
            
        Returns:
            The original object or an S3CheckpointFilter wrapping it.
        """
        if self.enabled and isinstance(o, TransformComponent) and not isinstance(o, DoNotCheckpoint):
            logger.debug(
                f'Wrapping with S3 checkpoint filter '
                f'[checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]'
            )
            return S3CheckpointFilter(
                checkpoint_name=self.checkpoint_name,
                bucket=self.bucket,
                prefix=self.prefix,
                inner=o,
                tenant_id=tenant_id,
            )
        else:
            logger.debug(
                f'Not wrapping with S3 checkpoint filter '
                f'[checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]'
            )
            return o
    
    def add_writer(self, o: Any) -> Any:
        """Add S3 checkpoint writer to a node handler.
        
        Wraps the provided node handler with an S3-based checkpoint writer
        if conditions are met (enabled, is NodeHandler).
        
        Args:
            o: The node handler to potentially wrap.
            
        Returns:
            The original object or an S3CheckpointWriter wrapping it.
        """
        if self.enabled and isinstance(o, NodeHandler):
            logger.debug(
                f'Wrapping with S3 checkpoint writer '
                f'[checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]'
            )
            return S3CheckpointWriter(
                checkpoint_name=self.checkpoint_name,
                bucket=self.bucket,
                prefix=self.prefix,
                inner=o,
            )
        else:
            logger.debug(
                f'Not wrapping with S3 checkpoint writer '
                f'[checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]'
            )
            return o
    
    def clear(self) -> int:
        """Clear all checkpoints for this checkpoint name from S3.
        
        Useful for restarting a workflow from scratch.
        
        Returns:
            Number of checkpoint objects deleted.
        """
        prefix = f"{self.prefix}/{SAVEPOINT_ROOT_DIR}/{self.checkpoint_name}/"
        deleted_count = 0
        
        try:
            paginator = GraphRAGConfig.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                    
                objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects_to_delete:
                    GraphRAGConfig.s3.delete_objects(
                        Bucket=self.bucket,
                        Delete={'Objects': objects_to_delete}
                    )
                    deleted_count += len(objects_to_delete)
            
            logger.info(f'Cleared {deleted_count} S3 checkpoints from s3://{self.bucket}/{prefix}')
            return deleted_count
            
        except ClientError as e:
            logger.error(f'Failed to clear S3 checkpoints: {e}')
            raise
    
    def count(self) -> int:
        """Count the number of checkpoints stored in S3.
        
        Returns:
            Number of checkpoint objects.
        """
        prefix = f"{self.prefix}/{SAVEPOINT_ROOT_DIR}/{self.checkpoint_name}/"
        count = 0
        
        try:
            paginator = GraphRAGConfig.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' in page:
                    count += len(page['Contents'])
            
            return count
            
        except ClientError as e:
            logger.error(f'Failed to count S3 checkpoints: {e}')
            raise
