# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional, Any, Tuple
from pydantic import ConfigDict, Field

from graphrag_toolkit.lexical_graph.retrieval.post_processors.reranker_mixin import (
    RerankerMixin,
)
from graphrag_toolkit.lexical_graph.retrieval.utils.statement_utils import (
    get_top_free_gpus,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as e:
    raise ImportError(
        "torch package not found, install with 'pip install torch'"
    ) from e


class BGEReranker(BaseNodePostprocessor, RerankerMixin):
    """
    Represents a Transformer-based reranker for node ranking tasks, built upon
    the FlagEmbedding reranker implementation. This class is designed to handle
    GPU-based re-ranking of node pairs using pre-trained language models, allowing
    for efficient and scalable processing in applications such as search relevance
    and text similarity scoring.

    The class leverages batch processing for efficient computation and includes
    methods to adjust configurations dynamically. It is optimized for single GPU
    usage and expects CUDA-compatible devices for execution.

    :ivar model_config: Configuration dictionary for managing model settings and namespaces.
    :type model_config: ConfigDict
    :ivar model_name: The name of the model used for reranking.
    :type model_name: str
    :ivar gpu_id: Optional ID of the GPU utilized for processing.
    :type gpu_id: Optional[int]
    :ivar reranker: The initialized reranker model instance.
    :type reranker: Any
    :ivar device: The CUDA device used for computation.
    :type device: Any
    :ivar batch_size_internal: Internal batch size used for input processing.
    :type batch_size_internal: int
    """

    model_config = ConfigDict(protected_namespaces=('model_validate', 'model_dump'))

    model_name: str = Field(default='BAAI/bge-reranker-v2-minicpm-layerwise')
    gpu_id: Optional[int] = Field(default=None)
    reranker: Any = Field(default=None)
    device: Any = Field(default=None)
    batch_size_internal: int = Field(default=128)

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-v2-minicpm-layerwise',
        gpu_id: Optional[int] = None,
        batch_size: int = 128,
    ):
        """
        Initializes the `BGEReranker` class, which configures a model for reranking
        utilizing GPU-based acceleration. The class requires a model name and device
        configuration for processing and uses the `FlagEmbedding` library for model
        initialization. This initialization ensures compatibility with supported
        GPUs and optimizes memory allocation as needed.

        :param model_name: Name of the reranker model to be loaded.
        :type model_name: str
        :param gpu_id: ID of the GPU to be used. If None, defaults to the most suitable GPU.
        :type gpu_id: Optional[int]
        :param batch_size: Number of samples to process in a batch.
        :type batch_size: int

        :raises ImportError: If `FlagEmbedding` library is not available.
        :raises Exception: If a GPU-compatible environment is not detected.

        """
        super().__init__()
        try:
            from FlagEmbedding import LayerWiseFlagLLMReranker
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding package not found, install with 'pip install FlagEmbedding'"
            ) from e
        self.model_name = model_name
        self.batch_size_internal = batch_size
        self.gpu_id = gpu_id

        try:
            if torch.cuda.is_available() and self.gpu_id is not None:
                self.device = torch.device(f'cuda:{self.gpu_id}')
            elif torch.cuda.is_available():
                self.gpu_id = get_top_free_gpus(n=1)[0]
                self.device = torch.device(f'cuda:{self.gpu_id}')
        except Exception:
            raise ("BGEReranker requires a GPU")

        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        try:
            self.reranker = LayerWiseFlagLLMReranker(
                model_name, use_fp16=True, devices=self.gpu_id, cutoff_layers=[28]
            )
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {str(e)}")
            raise

    @property
    def batch_size(self):
        """Gets the batch size for the internal configuration.

        This property retrieves the value of the batch size from the internal
        state of the object. It is commonly used to access the configured
        batch size for operations requiring batching functionality.

        Returns:
            int: The size of the batch currently configured internally.
        """
        return self.batch_size_internal

    def rerank_pairs(
        self, pairs: List[Tuple[str, str]], batch_size: int = 128
    ) -> List[float]:
        """Re-ranks a list of sentence pairs based on a pre-trained reranker
        model and returns the computed scores. This method utilizes a single
        GPU for computing the scores and is optimized for batch processing.

        Args:
            pairs (List[Tuple[str, str]]): A list of sentence pairs to be re-ranked.
                Each pair includes two sentences as strings.
            batch_size (int): The size of the batch for processing sentence pairs.
                Defaults to 128.

        Returns:
            List[float]: A list of scores corresponding to the input sentence pairs,
                where higher scores indicate higher relevance or similarity.

        Raises:
            Exception: If an error occurs during the re-ranking process, a detailed
                exception is logged and re-raised.
        """
        try:
            with torch.cuda.device(self.device):
                scores = self.reranker.compute_score_single_gpu(
                    sentence_pairs=pairs, batch_size=batch_size, cutoff_layers=[28]
                )
                return scores
        except Exception as e:
            logger.error(f"Error in rerank_pairs: {str(e)}")
            raise

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocesses a list of nodes by reranking them based on a query
        using a scoring mechanism.

        This method takes a list of nodes and an optional query bundle, calculates a relevance
        score for each node based on the provided query, and sorts the nodes based on the
        calculated scores. If the reranking process fails, it logs the error and returns
        the original list of nodes. Additionally, the method manages CUDA memory cache if a
        GPU is available.

        Args:
            nodes (List[NodeWithScore]): A list of nodes with initial scores to be processed.
            query_bundle (Optional[QueryBundle]): An optional query bundle containing the query
                string used for relevance scoring.

        Returns:
            List[NodeWithScore]: A list of nodes reranked by their calculated relevance scores.
            If reranking fails, the returned list will be the same as the input nodes.
        """
        if not query_bundle or not nodes:
            return nodes

        try:
            pairs = [(query_bundle.query_str, node.node.text) for node in nodes]

            scores = self.rerank_pairs(pairs, self.batch_size_internal)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            scored_nodes = [
                NodeWithScore(
                    node=node.node,
                    score=float(score) if isinstance(score, torch.Tensor) else score,
                )
                for node, score in zip(nodes, scores)
            ]

            scored_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            return scored_nodes

        except Exception as e:
            logger.error(f"BGE reranking failed: {str(e)}. Returning original nodes.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return nodes

    def __del__(self):
        """
        The '__del__' method ensures that GPU memory is cleared when an instance of the object is deleted.
        This is useful for managing resources in environments with limited GPU memory.

        :return: None
        """
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
