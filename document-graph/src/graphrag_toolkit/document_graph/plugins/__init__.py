# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugins — extension hooks for custom transform and validation steps.

document-graph uses pluggy to allow external packages to register custom
transformers, validators, and post-build hooks without modifying core code.

Usage:
    1. Define a plugin class with @hookimpl methods
    2. Register it with the PluginManager
    3. The pipeline calls hooks at each extension point

Extension Points:
    - pre_transform: modify records before transformation
    - post_transform: modify records after transformation
    - validate_node: validate a node before graph write
    - post_build: run logic after graph construction (e.g., metrics, alerts)
"""

import pluggy

PROJECT_NAME = "document_graph"

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class DocumentGraphHookSpec:
    """Hook specifications for document-graph plugin extension points."""

    @hookspec
    def pre_transform(self, records: list) -> list:
        """Called before transformation. Can modify or filter records.

        Args:
            records: List of record dicts from extraction.

        Returns:
            Modified list of records (or original if unchanged).
        """

    @hookspec
    def post_transform(self, records: list) -> list:
        """Called after transformation. Can enrich or validate transformed records.

        Args:
            records: List of transformed record dicts.

        Returns:
            Modified list of records.
        """

    @hookspec
    def validate_node(self, node_id: str, labels: list, properties: dict) -> bool:
        """Called before writing a node to the graph. Return False to skip.

        Args:
            node_id: The node's identifier.
            labels: List of node labels.
            properties: Node properties dict.

        Returns:
            True to allow write, False to skip this node.
        """

    @hookspec
    def post_build(self, stats: dict) -> None:
        """Called after graph build completes. For metrics, logging, alerts.

        Args:
            stats: Build statistics (nodes_written, edges_written, duration, etc.)
        """


def get_plugin_manager() -> pluggy.PluginManager:
    """Create and return a configured plugin manager.

    Returns:
        PluginManager with document-graph hookspecs registered.
    """
    pm = pluggy.PluginManager(PROJECT_NAME)
    pm.add_hookspecs(DocumentGraphHookSpec)
    return pm
