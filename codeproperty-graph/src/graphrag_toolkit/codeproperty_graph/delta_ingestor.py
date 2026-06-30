# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delta Ingestor — orchestrates skip-or-replace CPG ingestion."""

import logging
from datetime import datetime, timezone
from typing import Any

from .graph_diff import GraphDiff
from .manifest_manager import ManifestManager
from .models import Manifest
from .tenant_ops import delete_tenant

logger = logging.getLogger(__name__)


class DeltaIngestor:
    """Orchestrates CPG delta ingestion: compare → skip/ingest → purge → update manifest.

    Usage:
        ingestor = DeltaIngestor(bucket="<your-artifacts-bucket>")
        result = await ingestor.ingest(
            repo="amigo-core",
            job_id="uuid",
            nodes_data=[...],
            edges_data=[...],
            graph_store=neptune_store,
            write_fn=my_write_function,
        )
    """

    def __init__(self, bucket: str, prefix: str = "cpg-exports", region: str = "us-east-1"):
        self._manifest_mgr = ManifestManager(bucket, prefix, region)

    async def ingest(
        self,
        repo: str,
        job_id: str,
        tenant_id: str,
        nodes_data: list[dict],
        edges_data: list[dict],
        nodes_path: str,
        edges_path: str,
        graph_store: Any,
        write_fn=None,
    ) -> dict:
        """Execute delta-aware ingestion.

        Args:
            repo: Repository name
            job_id: Current job UUID
            tenant_id: Derived tenant for this job
            nodes_data: Parsed node records from Joern export
            edges_data: Parsed edge records from Joern export
            nodes_path: S3 URI to nodes.json
            edges_path: S3 URI to edges.json
            graph_store: Neptune graph store instance
            write_fn: async callable(nodes_data, edges_data, tenant_id, graph_store) → dict

        Returns:
            Dict with status, nodes_written, etc.
        """
        # Extract method signatures from nodes
        method_sigs = {
            n["full_name"]: n.get("hash", "")
            for n in nodes_data
            if n.get("node_type") == "METHOD" and n.get("full_name")
        }

        # Check manifest
        changed, previous = self._manifest_mgr.has_changes(repo, method_sigs)

        if not changed:
            logger.info(f"Delta check: no changes for {repo}, skipping ingest")
            return {
                "status": "SKIPPED",
                "reason": "no_changes",
                "tenant_id": previous.tenant_id,
                "previous_job_id": previous.job_id,
            }

        # Log diff if previous exists
        if previous:
            diff = GraphDiff.compare(previous.method_signatures, method_sigs)
            logger.info(f"Delta: {diff.summary} for {repo}")

        # Perform full ingest
        if write_fn:
            result = await write_fn(nodes_data, edges_data, tenant_id, graph_store)
        else:
            result = {"nodes_written": len(nodes_data), "edges_written": len(edges_data)}

        # Purge old tenant
        if previous and previous.tenant_id != tenant_id:
            try:
                await delete_tenant(previous.tenant_id, graph_store)
            except Exception:
                pass  # logged inside delete_tenant

        # Update manifest
        new_manifest = Manifest(
            repo=repo,
            signature=self._manifest_mgr.compute_signature(method_sigs),
            job_id=job_id,
            tenant_id=tenant_id,
            exported_at=datetime.now(timezone.utc).isoformat(),
            nodes_path=nodes_path,
            edges_path=edges_path,
            method_signatures=method_sigs,
        )
        self._manifest_mgr.put(new_manifest)

        result["status"] = "INGESTED"
        result["tenant_id"] = tenant_id
        result["delta"] = GraphDiff.compare(
            previous.method_signatures if previous else {}, method_sigs
        ).summary

        return result
