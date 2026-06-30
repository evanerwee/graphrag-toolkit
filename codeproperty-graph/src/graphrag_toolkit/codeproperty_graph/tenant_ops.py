# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tenant Operations — lifecycle management for CPG tenants in Neptune."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def delete_tenant(tenant_id: str, graph_store) -> int:
    """Purge all nodes and edges for a tenant from Neptune.

    Args:
        tenant_id: The tenant scope to delete
        graph_store: Neptune graph store with execute_query method

    Returns:
        Number of nodes deleted (approximate)
    """
    cypher = f"MATCH (n) WHERE n.tenant_id = '{tenant_id}' DETACH DELETE n"
    try:
        await asyncio.to_thread(graph_store.execute_query, cypher, {})
        logger.info(f"Tenant purged: {tenant_id}")
        return -1  # Neptune doesn't return count on DELETE
    except Exception as e:
        logger.warning(f"Tenant purge failed for {tenant_id}: {e}")
        raise
