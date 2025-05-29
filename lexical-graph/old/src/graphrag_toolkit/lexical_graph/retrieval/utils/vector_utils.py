# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
from typing import Optional

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)


def get_diverse_vss_elements(
    index_name: str,
    query_bundle: QueryBundle,
    vector_store: VectorStore,
    args: ProcessorArgs,
    filter_config: Optional[FilterConfig],
):
    """
    Retrieve a diverse set of elements from a vector search store (VSS) based on the specified top-k
    and diversity factor. This function ensures that elements are sampled from different sources
    based on their diversity.

    :param index_name: The name of the index to query.
    :type index_name: str
    :param query_bundle: The query information used to fetch results from the vector store.
    :type query_bundle: QueryBundle
    :param vector_store: The vector store to query for elements.
    :type vector_store: VectorStore
    :param args: Arguments containing settings for the query, including top-k and diversity factor.
    :type args: ProcessorArgs
    :param filter_config: Optional filter configuration to apply additional constraints on the query.
    :type filter_config: Optional[FilterConfig]
    :return: A list of diverse elements retrieved from the vector search store.
    :rtype: List[dict]
    """
    diversity_factor = args.vss_diversity_factor
    vss_top_k = args.vss_top_k

    if not diversity_factor or diversity_factor < 1:
        return vector_store.get_index(index_name).top_k(
            query_bundle, top_k=vss_top_k, filter_config=filter_config
        )

    top_k = vss_top_k * diversity_factor

    elements = vector_store.get_index(index_name).top_k(
        query_bundle, top_k=top_k, filter_config=filter_config
    )

    source_map = {}

    for element in elements:
        source_id = element['source']['sourceId']
        if source_id not in source_map:
            source_map[source_id] = queue.Queue()
        source_map[source_id].put(element)

    elements_by_source = queue.Queue()

    for source_elements in source_map.values():
        elements_by_source.put(source_elements)

    diverse_elements = []

    while (not elements_by_source.empty()) and len(diverse_elements) < vss_top_k:
        source_elements = elements_by_source.get()
        diverse_elements.append(source_elements.get())
        if not source_elements.empty():
            elements_by_source.put(source_elements)

    logger.debug(
        f'Diverse {index_name}s:\n'
        + '\n--------------\n'.join([str(element) for element in diverse_elements])
    )

    return diverse_elements
