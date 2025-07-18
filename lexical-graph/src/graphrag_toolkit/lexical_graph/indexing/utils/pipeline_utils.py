# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pipe import Pipe
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional, Sequence, Any, cast, Callable, Generator, Union


from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.core.schema import BaseNode, Document

def _sink():
    def _sink_from(generator):
        for item in generator:
            pass
    return Pipe(_sink_from)

sink = _sink()

def run_pipeline(
    pipeline:IngestionPipeline,
    node_batches:List[List[BaseNode]],
    cache_collection: Optional[str] = None,
    in_place: bool = True,
    num_workers: int = 1,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    transform: Callable[[List[BaseNode]], List[BaseNode]] = partial(
        run_transformations,
        transformations=pipeline.transformations,
        in_place=in_place,
        cache=pipeline.cache if not pipeline.disable_cache else None,
        cache_collection=cache_collection,
        **kwargs
    )

    with ProcessPoolExecutor(max_workers=num_workers) as p:
        processed_node_batches = p.map(transform, node_batches)
        processed_nodes = sum(processed_node_batches, start=cast(List[BaseNode], []))

    return processed_nodes

def node_batcher(
        num_batches: int, nodes: Union[Sequence[BaseNode], List[Document]]
    ) -> Generator[Union[Sequence[BaseNode], List[Document]], Any, Any]:
        num_nodes = len(nodes)
        batch_size = max(1, int(num_nodes / num_batches))
        if batch_size * num_batches < num_nodes:
             batch_size += 1
        for i in range(0, num_nodes, batch_size):
            yield nodes[i : i + batch_size]
