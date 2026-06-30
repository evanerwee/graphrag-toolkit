# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample plugins demonstrating document-graph extension points.

These plugins show how to hook into the pipeline without modifying core code.
Use them as templates for your own custom plugins.
"""

from datetime import datetime, timezone
from graphrag_toolkit.document_graph.plugins import hookimpl


class AuditTrailPlugin:
    """Adds audit metadata to every record passing through the pipeline.

    Stamps each record with:
    - _ingested_at: ISO timestamp of when the record was processed
    - _pipeline_version: version tag for lineage tracking
    """

    def __init__(self, pipeline_version: str = "1.0.0"):
        self.pipeline_version = pipeline_version

    @hookimpl
    def pre_transform(self, records: list) -> list:
        now = datetime.now(timezone.utc).isoformat()
        for record in records:
            record['_ingested_at'] = now
            record['_pipeline_version'] = self.pipeline_version
        return records


class DataQualityPlugin:
    """Validates records and rejects those that don't meet quality criteria.

    Checks:
    - Required fields are present and non-empty
    - No record exceeds max property count (prevents graph bloat)
    """

    def __init__(self, required_fields: list = None, max_properties: int = 50):
        self.required_fields = required_fields or []
        self.max_properties = max_properties
        self.rejected = []

    @hookimpl
    def pre_transform(self, records: list) -> list:
        valid = []
        for record in records:
            # Check required fields
            missing = [f for f in self.required_fields if not record.get(f)]
            if missing:
                self.rejected.append({'record': record, 'reason': f'missing fields: {missing}'})
                continue

            # Check property count
            if len(record) > self.max_properties:
                self.rejected.append({'record': record, 'reason': f'too many properties: {len(record)}'})
                continue

            valid.append(record)
        return valid


class NodeValidatorPlugin:
    """Validates nodes before graph write — reject nodes with empty IDs or blocked labels."""

    def __init__(self, blocked_labels: list = None):
        self.blocked_labels = blocked_labels or ['_Internal', '_System', '_Temp']
        self.skipped = []

    @hookimpl
    def validate_node(self, node_id: str, labels: list, properties: dict) -> bool:
        if not node_id or node_id.strip() == '':
            self.skipped.append({'node_id': node_id, 'reason': 'empty ID'})
            return False

        for label in labels:
            if label in self.blocked_labels:
                self.skipped.append({'node_id': node_id, 'reason': f'blocked label: {label}'})
                return False

        return True


class MetricsPlugin:
    """Collects build metrics for monitoring and alerting."""

    def __init__(self):
        self.build_history = []

    @hookimpl
    def post_build(self, stats: dict) -> None:
        stats['recorded_at'] = datetime.now(timezone.utc).isoformat()
        self.build_history.append(stats)

        # Alert on large builds
        nodes = stats.get('nodes_written', 0)
        if nodes > 10000:
            print(f"⚠️  Large build detected: {nodes} nodes written")

    def get_summary(self) -> dict:
        if not self.build_history:
            return {'builds': 0}
        total_nodes = sum(b.get('nodes_written', 0) for b in self.build_history)
        total_edges = sum(b.get('edges_written', 0) for b in self.build_history)
        return {
            'builds': len(self.build_history),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'last_build': self.build_history[-1],
        }
