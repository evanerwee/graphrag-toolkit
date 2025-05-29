# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import threading
import logging
from typing import Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore

logger = logging.getLogger(__name__)

def cosine_similarity(query_embedding, statement_embeddings):
    """
    Compute the cosine similarity between a query embedding and a set of statement
    embeddings.

    This function calculates the cosine similarity between a provided query
    embedding and a list of statement embeddings. The function returns a tuple
    consisting of similarity scores and the corresponding statement IDs. If the
    list of statement embeddings is empty, it returns an empty array and an empty
    list.

    Args:
        query_embedding: A 1D array-like structure representing the vector
            embedding of the query.
        statement_embeddings: A dictionary where keys are statement IDs and
            values are 1D array-like structures representing the vector embeddings
            of the corresponding statements.

    Returns:
        tuple: A tuple containing:
            similarities (numpy.ndarray): An array of cosine similarity scores
                between the query embedding and each of the statement embeddings.
            statement_ids (list): A list of IDs corresponding to the statement
                embeddings provided in the input.

    Raises:
        ValueError: If query_embedding or any of the values in
            statement_embeddings cannot be converted to a 1D numpy array.
        ZeroDivisionError: If any embedding has zero magnitude causing a division
            by zero during normalization.
    """
    if not statement_embeddings:
        return np.array([]), []

    query_embedding = np.array(query_embedding)
    statement_ids, statement_embeddings = zip(*statement_embeddings.items())
    statement_embeddings = np.array(statement_embeddings)

    dot_product = np.dot(statement_embeddings, query_embedding)
    norms = np.linalg.norm(statement_embeddings, axis=1) * np.linalg.norm(query_embedding)
    
    similarities = dot_product / norms
    return similarities, statement_ids

def get_top_k(query_embedding, statement_embeddings, top_k):
    """
    Fetches the top_k most similar statements to the given query embedding based on
    cosine similarity.

    Args:
        query_embedding: A vector that represents the query to compare against
            statement embeddings.
        statement_embeddings: A list of vectors where each vector represents a
            statement for similarity comparison. If an empty list is provided,
            the function returns an empty result.
        top_k: The number of top statements to retrieve based on cosine similarity.

    Returns:
        A list of tuples where each tuple contains the similarity score and the
        corresponding statement ID. The list is sorted in descending order of
        similarity scores. An empty list is returned if no similarities are
        calculated or if statement_embeddings is empty.
    """
    logger.debug(f'statement_embeddings: {statement_embeddings}')

    if not statement_embeddings:
        return []  
    
    similarities, statement_ids = cosine_similarity(query_embedding, statement_embeddings)

    logger.debug(f'similarities: {similarities}')
    
    if len(similarities) == 0:
        return []

    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    top_statement_ids = [statement_ids[idx] for idx in top_indices]
    top_similarities = similarities[top_indices]
    return list(zip(top_similarities, top_statement_ids))

def get_statements_query(graph_store, statement_ids):
    """
    Generates and executes a Cypher query to fetch detailed information about
    statements and their associated chunks and sources from the graph database.

    The function retrieves data such as statement IDs, chunk details, and source
    metadata for a list of given statement IDs. It ensures that the results are
    aligned with the requested statement IDs for accuracy.

    Args:
        graph_store: An instance of a graph database store that provides methods
            to handle node IDs and execute queries.
        statement_ids: A list of statement IDs for which the details are to be
            retrieved.

    Returns:
        list: A list of statement records where each record contains statement
            details, along with associated chunk and source information.
    """
    cypher = f'''
    MATCH (statement:`__Statement__`)-[:`__MENTIONED_IN__`]->(chunk:`__Chunk__`)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`) WHERE {graph_store.node_id("statement.statementId")} in $statement_ids
    RETURN {{
        {node_result('statement', graph_store.node_id("statement.statementId"))},
        source: {{
            sourceId: {graph_store.node_id("source.sourceId")},
            {node_result('source', key_name='metadata')}
        }},
        {node_result('chunk', graph_store.node_id("chunk.chunkId"))}
    }} AS result
    '''
    params = {'statement_ids': statement_ids}
    statements = graph_store.execute_query(cypher, params)
    results = []
    for statement_id in statement_ids:
                for statement in statements:
                    if statement['result']['statement']['statementId'] == statement_id:
                        results.append(statement)
    return results

def get_free_memory(gpu_index):
    """
    Retrieves the amount of free memory on a specific GPU device in megabytes (MB).
    This function uses the NVIDIA Management Library (NVML) to query the memory
    information of the GPU specified by its index.

    Args:
        gpu_index (int): The index of the GPU device for which the free memory
            needs to be retrieved.

    Returns:
        int: The amount of free memory on the specified GPU in megabytes (MB).

    Raises:
        ImportError: If the `pynvml` library is not installed, an ImportError is
            raised instructing the user to install the library.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free // 1024 ** 2
    except ImportError as e:
        raise ImportError(
                "pynvml package not found, install with 'pip install pynvml'"
            ) from e

def get_top_free_gpus(n=2):
    """
    Gets indices of the top `n` GPUs based on free memory in descending order.

    This function utilizes the PyTorch library to check the amount of free memory
    available on each GPU. It returns the indices of the GPUs with the highest
    free memory. If the PyTorch library is not installed, an ImportError will
    be raised, providing guidance on how to install it.

    Args:
        n (int, optional): The number of top GPUs to return based on free memory.
            Defaults to 2.

    Returns:
        list[int]: A list of integers representing the indices of GPUs with the
            most free memory.

    Raises:
        ImportError: If the torch package is not installed.
    """
    try:
        import torch
        free_memory = []
        for i in range(torch.cuda.device_count()):
            free_memory.append(get_free_memory(i))
        top_indices = sorted(range(len(free_memory)), key=lambda i: free_memory[i], reverse=True)[:n]
        return top_indices
    except ImportError as e:
        raise ImportError(
                "torch package not found, install with 'pip install torch'"
            ) from e

class SharedEmbeddingCache:
    """
    SharedEmbeddingCache manages a cache for embeddings, which are fetched from a
    vector store with retry logic. It ensures efficient retrieval of embeddings,
    minimizing repeated calls to the external data source.

    This class is designed to store and manage embeddings in a thread-safe manner.
    It keeps track of requested embeddings in cache, and when requested embeddings
    are missing, these are fetched and cached for future use. The class includes
    retry logic to handle transient failures during fetching operations.

    Attributes:
        vector_store (VectorStore): The vector store containing statement indexes
            and embeddings.
        _cache (Dict[str, np.ndarray]): A dictionary acting as an internal cache
            to store embeddings by their statement IDs.
        _lock (threading.Lock): A lock to ensure thread-safe access to the cache or
            updates.
    """
    def __init__(self, vector_store:VectorStore):
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.vector_store = vector_store

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),retry=retry_if_exception_type(Exception))
    def _fetch_embeddings(self, statement_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Fetches embeddings for the specified statement IDs from the vector store.

        Utilizes an exponential backoff retry mechanism to handle transient errors.
        This method interacts with a vector store backend to retrieve embeddings
        associated with given statement IDs. The returned embeddings are transformed
        and formatted into a dictionary mapping each statement ID to a corresponding
        NumPy array.

        Args:
            statement_ids (List[str]): A list of statement IDs for which to fetch
                embeddings.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are statement IDs and
                values are NumPy arrays of the corresponding embeddings.
        """
        embeddings = self.vector_store.get_index('statement').get_embeddings(statement_ids)
        return {
            e['statement']['statementId']: np.array(e['embedding']) 
            for e in embeddings
        }

    def get_embeddings(self, statement_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Retrieves the embeddings for the provided statement identifiers. The method first attempts
        to fetch embeddings from the internal cache. If any requested embeddings are not found in
        the cache, it tries to fetch them using an external method, updates the cache, and
        returns all the collected embeddings. If fetching new embeddings fails, the method
        returns only the embeddings retrieved from the cache.

        Args:
            statement_ids (List[str]): A list of statement identifiers for which embeddings are needed.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping each statement identifier to its
            corresponding embedding as a NumPy array.
        """
        missing_ids = []
        cached_embeddings = {}

        logger.debug(f'statement_ids: {statement_ids}')

        # Check cache first
        for sid in statement_ids:
            if sid in self._cache:
                cached_embeddings[sid] = self._cache[sid]
            else:
                missing_ids.append(sid)

        logger.debug(f'missing_ids: {missing_ids}')

        # Fetch missing embeddings with retry
        if missing_ids:
            try:
                new_embeddings = self._fetch_embeddings(missing_ids)
                with self._lock:
                    self._cache.update(new_embeddings)
                    cached_embeddings.update(new_embeddings)
            except Exception as e:
                logger.error(f"Failed to fetch embeddings after retries: {e}")
                # Return what we have from cache
                logger.warning(f"Returning {len(cached_embeddings)} cached embeddings out of {len(statement_ids)} requested")

        return cached_embeddings