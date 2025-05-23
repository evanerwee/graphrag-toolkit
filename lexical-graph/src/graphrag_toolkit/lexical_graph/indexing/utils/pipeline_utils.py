# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pipe import Pipe
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional, Sequence, Any, cast, Callable


from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.core.schema import BaseNode


def _sink():
    """
    Creates a pipeline stage that consumes and discards all items from a generator.

    The `_sink` function constructs a pipeline stage that connects to a generator
    and processes the emitted items. The `_sink_from` function is applied to any
    incoming generator to iterate through all its items without performing any
    operation, effectively discarding them. This stage is useful for terminating a
    pipeline when no further processing or collection of items is required.

    :return: A `Pipe` object that discards all items from the connected generator.
    :rtype: Pipe
    """
    def _sink_from(generator):
        for item in generator:
            pass

    return Pipe(_sink_from)


sink = _sink()


def run_pipeline(
    pipeline: IngestionPipeline,
    node_batches: List[List[BaseNode]],
    cache_collection: Optional[str] = None,
    in_place: bool = True,
    num_workers: int = 1,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """
    Executes a pipeline to process batches of nodes by applying the transformations
    specified in the pipeline. This function uses a process pool to parallelize
    the execution of the transformations across multiple workers.

    :param pipeline: The ingestion pipeline containing the transformations to
        be applied to the nodes and optional caching configuration.
    :type pipeline: IngestionPipeline
    :param node_batches: List of batches of nodes to be processed. Each batch is
        a list of BaseNode objects.
    :type node_batches: List[List[BaseNode]]
    :param cache_collection: Name of the cache collection to use, or None if no
        cache collection is specified.
    :type cache_collection: Optional[str]
    :param in_place: Flag to specify whether transformations should be applied
        in-place on the nodes. Defaults to True.
    :type in_place: bool
    :param num_workers: Number of parallel worker processes to use for
        transformations. Defaults to 1.
    :type num_workers: int
    :param kwargs: Additional arguments to be passed to the transformation
        functions.
    :type kwargs: Any
    :return: A sequence of processed nodes obtained after applying the
        transformations.
    :rtype: Sequence[BaseNode]
    """
    transform: Callable[[List[BaseNode]], List[BaseNode]] = partial(
        run_transformations,
        transformations=pipeline.transformations,
        in_place=in_place,
        cache=pipeline.cache if not pipeline.disable_cache else None,
        cache_collection=cache_collection,
        **kwargs,
    )

    with ProcessPoolExecutor(max_workers=num_workers) as p:
        processed_node_batches = p.map(transform, node_batches)
        processed_nodes = sum(processed_node_batches, start=cast(List[BaseNode], []))

    return processed_nodes
