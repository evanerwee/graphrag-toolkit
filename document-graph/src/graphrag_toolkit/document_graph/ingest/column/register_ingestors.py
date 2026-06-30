"""Register column ingestors — registers all column-level ingestor providers."""
# Register all column_level ingestors
from graphrag_toolkit.document_graph.ingest.ingestors_provider_registry import IngestorProviderRegistry
from graphrag_toolkit.document_graph.ingest.column.column_selector import ColumnSelectorProvider
from graphrag_toolkit.document_graph.ingest.column.column_renamer import ColumnRenamerProvider
from graphrag_toolkit.document_graph.ingest.column.column_reorder import ColumnReorderProvider
from graphrag_toolkit.document_graph.ingest.column.column_type_converter import ColumnTypeConverterProvider

# Register column_level ingestors
IngestorProviderRegistry.register("column_selector", ColumnSelectorProvider)
IngestorProviderRegistry.register("column_renamer", ColumnRenamerProvider)
IngestorProviderRegistry.register("column_reorder", ColumnReorderProvider)
IngestorProviderRegistry.register("column_type_converter", ColumnTypeConverterProvider)

