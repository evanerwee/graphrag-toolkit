# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import logging
import random
from typing import Sequence, List, Any, Optional

from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.indexing.extract.infer_config import (
    OnExistingClassifications,
)
from graphrag_toolkit.lexical_graph.indexing.extract.source_doc_parser import (
    SourceDocParser,
)
from graphrag_toolkit.lexical_graph.indexing.extract import (
    ScopedValueStore,
    DEFAULT_SCOPE,
)
from graphrag_toolkit.lexical_graph.indexing.constants import (
    DEFAULT_ENTITY_CLASSIFICATIONS,
)
from graphrag_toolkit.lexical_graph.indexing.prompts import (
    DOMAIN_ENTITY_CLASSIFICATIONS_PROMPT,
)

from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.bridge.pydantic import Field
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

DEFAULT_NUM_SAMPLES = 5
DEFAULT_NUM_ITERATIONS = 1


class InferClassifications(SourceDocParser):
    """
    Handles the classification of text nodes and documents leveraging domain-specific
    classifications through a language model. The class performs sampling, chunking,
    and classification tasks iteratively for better domain adaptation. It also
    handles existing classifications with multiple merge strategies and provides
    options for extensive customization, including splitter, prompt template, and LLM
    usage.

    :ivar classification_store: Storage mechanism for classifications.
    :type classification_store: ScopedValueStore
    :ivar classification_label: Label used for classification tagging.
    :type classification_label: str
    :ivar classification_scope: Scope to which the classifications are applied.
    :type classification_scope: str
    :ivar num_samples: Number of chunks to sample per iteration.
    :type num_samples: int
    :ivar num_iterations: Number of iterations to sample documents or chunks.
    :type num_iterations: int
    :ivar splitter: Optional chunk splitter handling sentence-level division.
    :type splitter: Optional[SentenceSplitter]
    :ivar llm: Language model used for text processing and classification.
    :type llm: Optional[LLMCache]
    :ivar prompt_template: Template string for defining prompt structure in the LLM.
    :type prompt_template: str
    :ivar default_classifications: List of default classifications used when none are inferred.
    :type default_classifications: List[str]
    :ivar merge_action: Strategy to handle merging or retaining existing classifications.
    :type merge_action: OnExistingClassifications
    """

    classification_store: ScopedValueStore = Field(description='Classification store')

    classification_label: str = Field(description='Classification label')

    classification_scope: str = Field(description='Classification scope')

    num_samples: int = Field(description='Number of chunks to sample per iteration')

    num_iterations: int = Field(description='Number times to sample documents')

    splitter: Optional[SentenceSplitter] = Field(description='Chunk splitter')

    llm: Optional[LLMCache] = Field(description='The LLM to use for extraction')

    prompt_template: str = Field(description='Prompt template')

    default_classifications: List[str] = Field('Default classifications')

    merge_action: OnExistingClassifications = Field(
        'Action to take if there are existing classifications'
    )

    def __init__(
        self,
        classification_store: ScopedValueStore,
        classification_label: str,
        classification_scope: Optional[str] = None,
        num_samples: Optional[int] = None,
        num_iterations: Optional[int] = None,
        splitter: Optional[SentenceSplitter] = None,
        llm: Optional[LLMCacheType] = None,
        prompt_template: Optional[str] = None,
        default_classifications: Optional[List[str]] = None,
        merge_action: Optional[OnExistingClassifications] = None,
    ):
        """
        Initializes an instance with configurations for classification tasks, including
        handling of label storage, scope definition, sample processing, text splitting,
        language model selection, and classification management.

        :param classification_store: The storage instance where classification-related
            values are scoped and maintained.
        :type classification_store: ScopedValueStore
        :param classification_label: The label used to categorize the classification
            values.
        :type classification_label: str
        :param classification_scope: The scope under which the classification task
            is performed. If None, a default classification scope is used.
        :type classification_scope: Optional[str]
        :param num_samples: The number of samples to process within the classification
            task. Defaults to a predefined number if not specified.
        :type num_samples: Optional[int]
        :param num_iterations: The number of iterations allowed for the classification
            process. Defaults to a predefined value if not set.
        :type num_iterations: Optional[int]
        :param splitter: An optional splitter instance used for segmenting text
            samples before classification.
        :type splitter: Optional[SentenceSplitter]
        :param llm: The language model used for classification processing.
            If not provided, a default language model or caching mechanism is applied.
        :type llm: Optional[LLMCacheType]
        :param prompt_template: The template string used for prompt generation.
            Defaults to a predefined template if not given.
        :type prompt_template: Optional[str]
        :param default_classifications: The default list of classifications to apply
            when no specific classifications are provided.
        :type default_classifications: Optional[List[str]]
        :param merge_action: The strategy for resolving existing classifications during
            updates. Defaults to retaining existing values.
        :type merge_action: Optional[OnExistingClassifications]

        """
        super().__init__(
            classification_store=classification_store,
            classification_label=classification_label,
            classification_scope=classification_scope or DEFAULT_SCOPE,
            num_samples=num_samples or DEFAULT_NUM_SAMPLES,
            num_iterations=num_iterations or DEFAULT_NUM_ITERATIONS,
            splitter=splitter,
            llm=(
                llm
                if llm and isinstance(llm, LLMCache)
                else LLMCache(
                    llm=llm or GraphRAGConfig.extraction_llm,
                    enable_cache=GraphRAGConfig.enable_cache,
                )
            ),
            prompt_template=prompt_template or DOMAIN_ENTITY_CLASSIFICATIONS_PROMPT,
            default_classifications=default_classifications
            or DEFAULT_ENTITY_CLASSIFICATIONS,
            merge_action=merge_action or OnExistingClassifications.RETAIN_EXISTING,
        )

        logger.debug(f'Prompt template: {self.prompt_template}')

    def _parse_classifications(self, response_text: str) -> Optional[List[str]]:
        """
        Parses and extracts domain-specific classifications from the provided response text.

        This function searches for XML-styled tags, specifically `<entity_classifications>...</entity_classifications>`,
        in the given response text and retrieves the classifications defined within the tags. Lines are stripped
        and confined to meaningful entries to eliminate empty or whitespace-only records.

        If classifications are successfully extracted, their count is logged for informational purposes.
        Otherwise, a warning is logged, indicating failure to parse classifications.

        :param response_text: The textual response containing potential domain-specific classifications
            enclosed in `<entity_classifications>` tags.
        :type response_text: str
        :return: A list of extracted classifications if any are successfully parsed; otherwise, an empty list.
        :rtype: Optional[List[str]]
        """
        pattern = r'<entity_classifications>(.*?)</entity_classifications>'
        match = re.search(pattern, response_text, re.DOTALL)

        classifications = []

        if match:
            classifications.extend(
                [
                    line.strip()
                    for line in match.group(1).strip().split('\n')
                    if line.strip()
                ]
            )

        if classifications:
            logger.info(
                f'Successfully parsed {len(classifications)} domain-specific classifications'
            )
            return classifications
        else:
            logger.warning(
                f'Unable to parse classifications from response: {response_text}'
            )
            return classifications

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Parses a sequence of nodes for domain-specific classifications and performs
        classification adaptation using a specified number of iterations and
        samples. The method applies an optional splitter to divide input nodes
        into smaller chunks, processes these chunks with a language learning model,
        and retrieves or merges domain-specific classifications based on the
        defined action (retain existing, merge with existing, or use default).
        If successful, these classifications are saved; otherwise, default
        classifications are used.

        :param nodes:
            A sequence of nodes to analyze and classify.
        :param show_progress:
            A flag indicating whether to display progress
            during classification adaptation.
        :param kwargs:
            Additional arguments that may be needed
            for domain-specific implementations.
        :return:
            A list of nodes, potentially modified
            with classification metadata.
        """
        current_values = self.classification_store.get_scoped_values(
            self.classification_label, self.classification_scope
        )
        if (
            current_values
            and self.merge_action == OnExistingClassifications.RETAIN_EXISTING
        ):
            logger.info(
                f'Domain-specific classifications already exist [label: {self.classification_label}, scope: {self.classification_scope}, classifications: {current_values}]'
            )
            return nodes

        chunks = self.splitter(nodes) if self.splitter else nodes

        classifications = set()

        for i in range(1, self.num_iterations + 1):

            sample_chunks = (
                random.sample(chunks, self.num_samples)
                if len(chunks) > self.num_samples
                else chunks
            )

            logger.info(
                f'Analyzing {len(sample_chunks)} chunks for domain adaptation [iteration: {i}, merge_action: {self.merge_action}]'
            )

            formatted_chunks = '\n'.join(
                f'<chunk>{chunk.text}</chunk>' for chunk in sample_chunks
            )

            response = self.llm.predict(
                PromptTemplate(self.prompt_template), text_chunks=formatted_chunks
            )

            classifications.update(self._parse_classifications(response))

        if (
            current_values
            and self.merge_action == OnExistingClassifications.MERGE_EXISTING
        ):
            classifications.update(current_values)

        classifications = list(classifications)

        if classifications:
            logger.info(
                f'Domain adaptation succeeded [label: {self.classification_label}, scope: {self.classification_scope}, classifications: {classifications}]'
            )
            self.classification_store.save_scoped_values(
                self.classification_label, self.classification_scope, classifications
            )
        else:
            logger.warning(
                f'Domain adaptation failed, using default classifications [label: {self.classification_label}, scope: {self.classification_scope}, classifications: {self.default_classifications}]'
            )
            self.classification_store.save_scoped_values(
                self.classification_label,
                self.classification_scope,
                self.default_classifications,
            )

        return nodes

    def _parse_source_docs(self, source_documents):
        """
        Parses the provided source documents by iterating over their nodes,
        and processes the nodes through an internal method. This helps in
        breaking down the documents and organizing their components for
        further operations or processing.

        :param source_documents: A list of source documents to be parsed.
            Each document contains multiple nodes that are extracted and
            used in subsequent processing.
        :type source_documents: list
        :return: The list of processed source documents after parsing their nodes.
        :rtype: list
        """
        source_docs = [source_doc for source_doc in source_documents]

        nodes = [n for sd in source_docs for n in sd.nodes]

        self._parse_nodes(nodes)

        return source_docs
