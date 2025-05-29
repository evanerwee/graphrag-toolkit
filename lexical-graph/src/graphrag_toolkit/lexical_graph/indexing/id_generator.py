# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import Optional

from graphrag_toolkit.lexical_graph import TenantId

from llama_index.core.bridge.pydantic import BaseModel, Field

class IdGenerator(BaseModel):
    """
    A class responsible for generating unique and tenant-specific identifiers.

    The `IdGenerator` class is designed to create various types of identifiers,
    such as source IDs, chunk IDs, node IDs, and tenant-specific rewritten IDs.
    It uses hashing techniques for creating compact and unique representations
    of input values. The class also integrates with a tenant system to ensure
    that identifiers are applied in the context of a particular tenant.

    Attributes:
        tenant_id (TenantId): The tenant context that is used for generating
            tenant-specific IDs and rewriting ID values.
    """
    tenant_id:TenantId
    
    def __init__(self, tenant_id:TenantId=None):
        super().__init__(tenant_id=tenant_id or TenantId())

    def _get_hash(self, s):
        """
        Generates an MD5 hash for a given string.

        This private method computes the MD5 hash of a provided string and returns its
        hexadecimal representation. It is used internally to generate unique hashed
        values based on string inputs.

        Args:
            s: The input string to be hashed.

        Returns:
            The hexadecimal representation of the MD5 hash of the input string.
        """
        return hashlib.md5(s.encode('utf-8')).digest().hex()

    def create_source_id(self, text:str, metadata_str:str):
        """
        Generates a unique source identifier by combining hashed representations of a text
        and its associated metadata.

        This function is part of an identifier creation mechanism that hashes input strings
        and formats the resulting substrings into a specific pattern for source identification.
        The creation of the source ID involves truncating the hashed values to predefined lengths
        for compactness and uniqueness.

        Args:
            text: The primary content or body for which the source identifier is generated.
            metadata_str: Additional metadata or descriptive information related to the primary content.

        Returns:
            str: A formatted string representing the unique source identifier, combining
            hashed substrings derived from the input text and metadata.

        """
        return f"aws::{self._get_hash(text)[:8]}:{self._get_hash(metadata_str)[:4]}"
        
    def create_chunk_id(self, source_id:str, text:str, metadata_str:str):
        """
        Generates a unique chunk identifier by combining a source ID, a hash of the given text,
        and associated metadata.

        Args:
            source_id (str): The identifier of the source content.
            text (str): The primary content or text that needs to be identified.
            metadata_str (str): The metadata string used for constructing the identifier.

        Returns:
            str: A uniquely generated chunk identifier based on the given inputs.
        """
        return f'{source_id}:{self._get_hash(text + metadata_str)[:8]}'
    
    def rewrite_id_for_tenant(self, id_value:str):
        """
        Rewrites the provided ID with the tenant-specific ID format.

        This method utilizes the tenant's specific implementation of ID
        rewriting to convert the given ID to a tenant-specific format.

        Args:
            id_value: The original ID to be rewritten.

        Returns:
            str: The tenant-specific rewritten ID.
        """
        return self.tenant_id.rewrite_id(id_value)

    def create_node_id(self, node_type:str, v1:str, v2:Optional[str]=None) -> str:
        """
        Creates a unique identifier for a specific node based on the provided parameters.

        The function generates a hashable string using the `node_type`, `v1`, and
        optionally `v2` parameters. The parameters are formatted to lower case and
        spaces are replaced with underscores for consistency. If `v2` is provided,
        it is included in the resulting identifier; otherwise, it is excluded.

        Args:
            node_type: A string representing the type of the node.
            v1: A string specifying the first variable or identifier in the node's identity.
            v2: An optional string specifying the second variable or identifier
                in the node's identity.

        Returns:
            A string containing a unique hash-based identifier for the node.
        """
        if v2:
            return self._get_hash(self.tenant_id.format_hashable(f"{node_type.lower()}::{v1.lower().replace(' ', '_')}::{v2.lower().replace(' ', '_')}"))
        else:
            return self._get_hash(self.tenant_id.format_hashable(f"{node_type.lower()}::{v1.lower().replace(' ', '_')}"))