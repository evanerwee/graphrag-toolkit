# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import Optional

from graphrag_toolkit.lexical_graph import TenantId

from llama_index.core.bridge.pydantic import BaseModel, Field


class IdGenerator(BaseModel):
    """
    This class provides functionality to generate various unique identifiers,
    tailored for tenant-specific operations. It centralizes the creation of hash-based
    identifiers, node identifiers, and tenant-specific rewrites of IDs.

    The class incorporates methods for creating source and chunk identifiers using
    text and metadata inputs, as well as methods for tenant-specific operations
    like rewriting IDs or generating node identifiers. It heavily relies on hashing
    mechanisms to ensure uniqueness and compactness of the generated identifiers.

    :ivar tenant_id: Represents the unique tenant identifier associated with
        the object. It determines tenant-specific ID formatting and rewriting.
    :type tenant_id: TenantId
    """

    tenant_id: TenantId

    def __init__(self, tenant_id: TenantId = None):
        """
        Initializes a new instance of the class with a given tenant ID.

        :param tenant_id: Identifies the tenant. Defaults to a new instance
                          of the TenantId class if not provided.
        :type tenant_id: TenantId
        """
        super().__init__(tenant_id=tenant_id or TenantId())

    def _get_hash(self, s):
        """
        Compute the MD5 hash of a given string and return its hexadecimal digest.

        The '_get_hash' function takes a string input, encodes it in UTF-8,
        computes its MD5 hash, and returns the resulting value as a hexadecimal
        representation. This can be useful for ensuring data integrity, creating
        unique keys, or verifying content.

        :param s: The input string to be hashed.
        :type s: str
        :return: The hexadecimal MD5 digest of the input string.
        :rtype: str
        """
        return hashlib.md5(s.encode('utf-8')).digest().hex()

    def create_source_id(self, text: str, metadata_str: str):
        """
        Generates a unique source identifier based on the hashes of the given text
        and metadata strings. This method combines portions of the hashed values
        from the provided input strings to create the source ID in a specific format.

        :param text: The main text input to be used as part of the source ID.
        :type text: str
        :param metadata_str: Additional metadata to be included in the source ID
            generation.
        :type metadata_str: str
        :return: A formatted string representing the unique source identifier
            generated from the text and metadata inputs.
        :rtype: str
        """
        return f"aws::{self._get_hash(text)[:8]}:{self._get_hash(metadata_str)[:4]}"

    def create_chunk_id(self, source_id: str, text: str, metadata_str: str):
        """
        Generate a unique chunk identifier based on the provided source identifier, text, and metadata string.

        This method generates a unique identifier for a chunk by combining the given
        source ID, the concatenated content of `text` and `metadata_str`, and hashing
        their combination. The resulting identifier ensures uniqueness while preserving
        a consistent format.

        :param source_id: A string representing the identifier of the source.
        :param text: The chunk's text content, used as part of the identifier.
        :param metadata_str: Metadata related to the chunk, included in the hash.
        :return: A string representing the unique chunk identifier.
        """
        return f'{source_id}:{self._get_hash(text + metadata_str)[:8]}'

    def rewrite_id_for_tenant(self, id_value: str):
        """
        Rewrites the given ID for the tenant associated with the current instance. This method
        utilizes the tenant's rewrite_id functionality to return the adjusted ID.

        :param id_value: The original ID string that needs to be rewritten for the tenant.
        :type id_value: str
        :return: The rewritten ID string adjusted for the tenant.
        :rtype: str
        """
        return self.tenant_id.rewrite_id(id_value)

    def create_node_id(self, node_type: str, v1: str, v2: Optional[str] = None) -> str:
        """
        Generates a unique identifier for a node based on its type and associated values.
        The method creates a formatted string that combines the provided `node_type`,
        `v1`, and optionally `v2`, lowercase and spaces replaced by underscores.
        It then computes a hash of the formatted string using tenant-specific hashing logic.

        :param node_type: Represents the type of the node, case insensitive.
        :param v1: Primary identifier or component for the node.
        :param v2: Optional secondary identifier or component for the node, which adds
            specificity to the generated node ID if provided.
        :return: A hash string representing the unique node identifier.
        """
        if v2:
            return self._get_hash(
                self.tenant_id.format_hashable(
                    f"{node_type.lower()}::{v1.lower().replace(' ', '_')}::{v2.lower().replace(' ', '_')}"
                )
            )
        else:
            return self._get_hash(
                self.tenant_id.format_hashable(
                    f"{node_type.lower()}::{v1.lower().replace(' ', '_')}"
                )
            )
