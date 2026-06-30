# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register constructors — auto-registers all constructor providers with the registry."""

# Register all constructor providers
from graphrag_toolkit.document_graph.graph_build.constructors.constructors_provider_registry import ConstructorProviderRegistry

# Core constructors
from graphrag_toolkit.document_graph.graph_build.constructors.core.schema_driven_constructor import SchemaDrivenConstructor
from graphrag_toolkit.document_graph.graph_build.constructors.core.node_constructor import NodeConstructor
from graphrag_toolkit.document_graph.graph_build.constructors.core.edge_constructor import EdgeConstructor
from graphrag_toolkit.document_graph.graph_build.constructors.core.property_constructor import PropertyConstructor

# Pattern constructors
from graphrag_toolkit.document_graph.graph_build.constructors.patterns.one_to_many_constructor import OneToManyConstructor
from graphrag_toolkit.document_graph.graph_build.constructors.patterns.many_to_many_constructor import ManyToManyConstructor

# Optimization constructors
from graphrag_toolkit.document_graph.graph_build.constructors.optimizations.batch_constructor import BatchConstructor
from graphrag_toolkit.document_graph.graph_build.constructors.optimizations.deduplication_constructor import DeduplicationConstructor

# Register core constructors
ConstructorProviderRegistry.register("schema_driven", SchemaDrivenConstructor)
ConstructorProviderRegistry.register("node_constructor", NodeConstructor)
ConstructorProviderRegistry.register("edge_constructor", EdgeConstructor)
ConstructorProviderRegistry.register("property_constructor", PropertyConstructor)

# Register pattern constructors
ConstructorProviderRegistry.register("one_to_many", OneToManyConstructor)
ConstructorProviderRegistry.register("many_to_many", ManyToManyConstructor)

# Register optimization constructors
ConstructorProviderRegistry.register("batch_constructor", BatchConstructor)
ConstructorProviderRegistry.register("deduplication", DeduplicationConstructor)
