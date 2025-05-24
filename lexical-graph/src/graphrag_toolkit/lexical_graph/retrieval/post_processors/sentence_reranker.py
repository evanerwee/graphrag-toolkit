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
    A class used for reranking sentence pairs using a pretrained cross-encoder model.

    This class combines the functionalities of `SentenceTransformerRerank` and
    `RerankerMixin` to enable efficient reranking of sentence pairs based on their
    similarity scores. It provides configurations for top-n reranking, model selection,
    device setup, and batch processing. The primary use case is to compute relevance scores
    for a list of sentence pairs and retrieve the most relevant ones according to the
    given model.

    :ivar batch_size_internal: Internal variable to store the batch size for processing sentences.
    :type batch_size_internal: int
    """

    batch_size_internal: int = Field(default=128)

    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/stsb-distilroberta-base",
        device: Optional[str] = None,
        keep_retrieval_score: Optional[bool] = False,
        batch_size: Optional[int] = 128,
        **kwargs: Any
    ):
        """
        Initializes the instance of a class with the specified configurations. This constructor
        is responsible for importing necessary dependencies and setting up class attributes
        based on the provided arguments.

        :param top_n: The top number of results to consider for the operation.
        :type top_n: int
        :param model: The identifier of the model to load and utilize.
        :type model: str
        :param device: The device on which the computations will be performed. It can be set to a
            specific device like 'cpu' or 'cuda'.
        :type device: Optional[str]
        :param keep_retrieval_score: Determines whether to retain the retrieval score as part
            of the result output if applicable.
        :type keep_retrieval_score: Optional[bool]
        :param batch_size: Specifies the size of batches to be utilized during processing.
        :type batch_size: Optional[int]
        :param kwargs: Arbitrary keyword arguments to customize behavior.
        :type kwargs: Any
        :raises ImportError: If required packages `torch` and/or `sentence_transformers` are not found.
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

        self.batch_size_internal = batch_size

    @property
    def batch_size(self):
        """
        Property to get the batch size.

        This attribute returns the internal batch size value stored in
        the object. The property provides a way to retrieve the
        value without directly accessing the internal attribute.

        :raises AttributeError: If the internal batch size attribute
            does not exist or cannot be accessed.

        :return: The current internal batch size value.
        :rtype: int
        """
        return self.batch_size_internal

    def rerank_pairs(
        self, pairs: List[Tuple[str, str]], batch_size: int = 128
    ) -> List[float]:
        """
        Re-ranks a list of sentence pairs using the model by predicting a similarity
        score for each pair. The similarity scores are computed in batches of a
        specified size, which can improve efficiency for large input data.

        :param pairs: A list of sentence pairs for which similarity scores need to be
            computed. Each sentence pair is represented as a tuple of two strings.
        :param batch_size: The number of sentence pairs to be processed in a single
            batch. Defaults to 128.
        :return: A list of float values, where each value represents the computed
            similarity score for the corresponding sentence pair in the input list.
        """
        return self._model.predict(
            sentences=pairs, batch_size=batch_size, show_progress_bar=False
        )
