# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import abc

from typing import List
from llama_index.core.schema import BaseNode
from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.build_filters import BuildFilters
from graphrag_toolkit.lexical_graph.metadata import DefaultSourceMetadataFormatter

class DummyNodeBuilder(NodeBuilder):
    @classmethod
    def name(cls) -> str:
        return "DummyNodeBuilder"

    @classmethod
    def metadata_keys(cls) -> List[str]:
        return []
    
    def __init__(self):
        super().__init__(
            id_generator=IdGenerator(),
            build_filters=BuildFilters(),
            source_metadata_formatter=DefaultSourceMetadataFormatter()
        )

    def build_nodes(self, nodes:List[BaseNode], **kwargs) -> List[BaseNode]:
        return []


def test_puts_collection_based_metadata_items_into_invalid_metadata_section():

    builder = DummyNodeBuilder()

    metadata = {
        'id': 1,
        'name': 'my-name',
        'list-prop': [1, 2, 3],
        'dict-prop': {'a': 'A', 'b': 'B'},
        'set-prop': {'a', 'b', 'c'}
    }

    results = builder._get_source_info_metadata(metadata)

    assert 'metadata' in results
    assert 'invalid_metadata' in results
    assert len(results['metadata']) == 2
    assert len(results['invalid_metadata']) == 3
    assert 'id' in results['metadata']
    assert 'name' in results['metadata']
    assert 'list-prop' in results['invalid_metadata']
    assert 'dict-prop' in results['invalid_metadata']
    assert 'set-prop' in results['invalid_metadata']