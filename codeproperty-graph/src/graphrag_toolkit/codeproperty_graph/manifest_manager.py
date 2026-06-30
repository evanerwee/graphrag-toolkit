# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest Manager — S3-backed CPG state tracking."""

import hashlib
import json
import logging
from typing import Optional

import boto3

from .models import Manifest

logger = logging.getLogger(__name__)


class ManifestManager:
    """Read/write/compare CPG manifests on S3."""

    def __init__(self, bucket: str, prefix: str = "cpg-exports", region: str = "us-east-1"):
        self._bucket = bucket
        self._prefix = prefix
        self._s3 = boto3.client("s3", region_name=region)

    def compute_signature(self, method_signatures: dict[str, str]) -> str:
        """Compute sha256 signature from method full_name:hash pairs."""
        payload = json.dumps(method_signatures, sort_keys=True)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def get(self, repo: str) -> Optional[Manifest]:
        """Read manifest for a repo from S3. Returns None if not found."""
        key = f"{self._prefix}/{repo}/manifest.json"
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(resp["Body"].read())
            return Manifest(**data)
        except Exception:
            return None

    def put(self, manifest: Manifest) -> None:
        """Write manifest to S3."""
        key = f"{self._prefix}/{manifest.repo}/manifest.json"
        body = json.dumps({
            "repo": manifest.repo,
            "signature": manifest.signature,
            "job_id": manifest.job_id,
            "tenant_id": manifest.tenant_id,
            "exported_at": manifest.exported_at,
            "nodes_path": manifest.nodes_path,
            "edges_path": manifest.edges_path,
            "method_signatures": manifest.method_signatures,
        })
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body.encode(), ContentType="application/json")
        logger.info(f"Manifest written: s3://{self._bucket}/{key}")

    def has_changes(self, repo: str, current_signatures: dict[str, str]) -> tuple[bool, Optional[Manifest]]:
        """Check if current export differs from stored manifest.

        Returns:
            (has_changes: bool, previous_manifest: Optional[Manifest])
        """
        previous = self.get(repo)
        if not previous:
            return True, None
        new_sig = self.compute_signature(current_signatures)
        return new_sig != previous.signature, previous
