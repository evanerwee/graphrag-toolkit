# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple, Optional, Any

from graphrag_toolkit.lexical_graph.retrieval.post_processors import RerankerMixin

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor import SentenceTransformerRerank

logger = logging.getLogger(__name__)

class SentenceReranker(SentenceTransformerRerank, RerankerMixin):
    """
    Represents a specialized sentence reranker that combines functionalities from
    SentenceTransformerRerank and RerankerMixin.

    This class is designed to rerank sentence pairs using a pre-trained cross-encoder
    model. Users can specify parameters such as the top N results to rerank, the
    underlying model to use, and batch size. The class initializes with support
    for GPU execution if available and relies on external libraries such as
    `sentence_transformers` and `torch`.

    Attributes:
        batch_size_internal (int): Internal batch size used during reranking.
    """
    batch_size_internal: int = Field(default=128)

    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/stsb-distilroberta-base",
        device: Optional[str] = None,
        keep_retrieval_score: Optional[bool] = False,
        batch_size:Optional[int]=128,
        **kwargs:Any
    ):
        """
        Initializes the class with configuration for a model execution environment.

        This constructor sets up the initial parameters for handling model operations, such
        as determining the top number of results, model name, device configuration, batch size,
        and additional keyword arguments. It ensures that the required external dependencies
        are available and imports them upon initialization.

        Args:
            top_n (int): The number of top results to retrieve. Default is 2.
            model (str): The name or identifier of the model to be used.
                Default is "cross-encoder/stsb-distilroberta-base".
            device (Optional[str]): Specifies the device to execute the model.
                For example, 'cpu' or 'cuda'. Default is None.
            keep_retrieval_score (Optional[bool]): Determines whether to retain
                the retrieval score after execution. Default is False.
            batch_size (Optional[int]): Defines the size of batches to be processed
                during execution. Default is 128.
            **kwargs (Any): Additional keyword arguments for further customization
                of the model initialization.

        Raises:
            ImportError: If the required 'torch' and/or 'sentence_transformers'
                packages are not installed.
        """
        try:
            import sentence_transformers
            import torch
        except ImportError as e:
            raise ImportError(
                "torch and/or sentence_transformers packages not found, install with 'pip install torch sentence_transformers'"
            ) from e
        
        super().__init__(
            top_n=top_n,
            model=model,
            device=device,
            keep_retrieval_score=keep_retrieval_score, 
        )
        
        self.batch_size_internal=batch_size

    @property
    def batch_size(self):
        """
        Returns the internal batch size value.

        This property provides access to the internal variable `batch_size_internal`,
        which stores the batch size for the current object. It is used to retrieve the
        value externally without directly exposing the internal variable.

        Returns:
            int: The batch size value stored in the internal variable.
        """
        return self.batch_size_internal
    
    def rerank_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 128
    ) -> List[float]:
        """
        Re-ranks pairs of sentences based on their similarity or relevance by passing them
        through the model for prediction. It processes the pairs in batches for efficiency.

        Args:
            pairs (List[Tuple[str, str]]): A list of sentence pairs, where each pair consists
                of two strings to be compared.
            batch_size (int): The number of sentence pairs to process in a single batch.
                Defaults to 128.

        Returns:
            List[float]: A list of prediction scores for each pair, indicating their similarity
                or relevance.
        """
        return self._model.predict(sentences=pairs, batch_size=batch_size, show_progress_bar=False)

