# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import asyncio
from typing import List, Optional, Sequence, Dict

from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.indexing.model import Propositions
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY
from graphrag_toolkit.lexical_graph.indexing.prompts import EXTRACT_PROPOSITIONS_PROMPT

from llama_index.core.schema import BaseNode
from llama_index.core.bridge.pydantic import Field
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs


logger = logging.getLogger(__name__)


class LLMPropositionExtractor(BaseExtractor):
    """
    Handles the extraction of logical or semantic propositions from textual data
    using a language model backend.

    The `LLMPropositionExtractor` class is designed to process a sequence of nodes,
    extract propositions from their textual content, and return results in a structured
    format. It leverages an external language model configured with a prompt template
    to perform extraction tasks. The class supports asynchronous workflows, parallel
    processing, and customizable configurations. It is particularly suited for working
    with graph-based systems and metadata where organized extraction is required.

    :ivar llm: The LLM cache or configuration used for proposition extraction.
    :type llm: Optional[LLMCache]
    :ivar prompt_template: Template used to guide the LLM's response during proposition extraction.
    :type prompt_template: str
    :ivar source_metadata_field: Key in the node's metadata to retrieve text for extraction, if available.
    :type source_metadata_field: Optional[str]
    """

    llm: Optional[LLMCache] = Field(description='The LLM to use for extraction')

    prompt_template: str = Field(description='Prompt template')

    source_metadata_field: Optional[str] = Field(
        description='Metadata field from which to extract propositions'
    )

    @classmethod
    def class_name(cls) -> str:
        """
        Provides a class-level method to return the name of the class.

        The method does not take any additional parameters other than the
        mandatory `cls` parameter for the class itself. It returns a string
        representation of the class name and can be used universally across
        different instances or contexts of the class to get a consistent human-
        readable identifier for the class.

        :returns: A string that represents the name of the class.
        :rtype: str
        """
        return 'LLMPropositionExtractor'

    def __init__(
        self,
        llm: LLMCacheType = None,
        prompt_template=None,
        source_metadata_field=None,
        num_workers: Optional[int] = None,
    ):
        """
        Initializes an instance with the specified parameters and configurations. This constructor
        sets up the instance by initializing inherited properties and configuring the prompt template,
        the source metadata field, and the number of workers. If not explicitly provided, default
        values are utilized.

        :param llm: An instance of LLMCacheType used for extraction. Defaults to None, which results
           in the usage of GraphRAGConfig.extraction_llm. If `enable_cache` is configured, caching
           behaviors will also be activated.
        :param prompt_template: The template to be used for extracting propositions. If not
           provided, defaults to `EXTRACT_PROPOSITIONS_PROMPT`.
        :param source_metadata_field: The field from which source metadata will be extracted. If not
           provided, its value remains None.
        :param num_workers: The number of threads to be used for extraction. If not specified, it
           defaults to the value defined in `GraphRAGConfig.extraction_num_threads_per_worker`.

        """
        super().__init__(
            llm=(
                llm
                if llm and isinstance(llm, LLMCache)
                else LLMCache(
                    llm=llm or GraphRAGConfig.extraction_llm,
                    enable_cache=GraphRAGConfig.enable_cache,
                )
            ),
            prompt_template=prompt_template or EXTRACT_PROPOSITIONS_PROMPT,
            source_metadata_field=source_metadata_field,
            num_workers=num_workers or GraphRAGConfig.extraction_num_threads_per_worker,
        )

        logger.debug(f'Prompt template: {self.prompt_template}')

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """
        Asynchronously extracts propositions for a given list of nodes by processing
        them through an internal method. This method retrieves propositions specific
        to each node and compiles them into a list of dictionaries.

        :param nodes: A sequence of BaseNode objects for which propositions need to
            be extracted.
        :return: A list of dictionaries, where each dictionary represents the
            proposition entries extracted for the nodes.
        """
        proposition_entries = await self._extract_propositions_for_nodes(nodes)
        return [proposition_entry for proposition_entry in proposition_entries]

    async def _extract_propositions_for_nodes(self, nodes):
        """
        Asynchronously extracts propositions for the provided nodes.

        This method orchestrates the extraction of propositions for multiple nodes in parallel
        by creating and managing asynchronous jobs. The number of workers and whether a progress
        indicator is shown are configurable.

        :param nodes: A list of nodes for which propositions are to be extracted.
        :type nodes: list
        :return: A list of results from the proposition extraction process for each node.
        :rtype: list
        """
        jobs = [self._extract_propositions_for_node(node) for node in nodes]
        return await run_jobs(
            jobs,
            show_progress=self.show_progress,
            workers=self.num_workers,
            desc=f'Extracting propositions [nodes: {len(nodes)}, num_workers: {self.num_workers}]',
        )

    async def _extract_propositions_for_node(self, node):
        """
        Extracts propositions for a given node by processing its metadata or text content.

        This function uses the node's metadata or text to extract propositions.
        It utilizes an asynchronous process to fetch the proposition collection.
        Debug logging includes the input text and the resulting propositions.

        :param node: The node for which propositions need to be extracted.
        :type node: Node
        :return: A dictionary containing the extracted propositions.
        :rtype: dict
        """
        logger.debug(f'Extracting propositions for node {node.node_id}')
        text = (
            node.metadata.get(self.source_metadata_field, node.text)
            if self.source_metadata_field
            else node.text
        )
        proposition_collection = await self._extract_propositions(text)
        if logger.isEnabledFor(logging.DEBUG):
            s = f"""====================================
text: {text}
------------------------------------
propositions: {proposition_collection}
"""
            logger.debug(s)

        return {PROPOSITIONS_KEY: proposition_collection.model_dump()['propositions']}

    async def _extract_propositions(self, text):
        """
        Extracts and processes propositions from the provided text using an LLM call.

        This method makes an async call to a blocking function that interacts with the LLM.
        The response is processed to extract individual propositions, remove duplicates,
        and return them in a structured format.

        :param text: The input text for extracting propositions.
        :type text: str
        :return: A Propositions object containing a list of unique propositions extracted
                 from the input text.
        :rtype: Propositions
        """

        def blocking_llm_call():
            """
            LLMPropositionExtractor is a custom extractor leveraging an LLM (Language Model)
            to asynchronously extract propositions based on a provided prompt template.
            It utilizes an underlying base LLM component to perform predictions
            and parses textual input to generate the relevant propositions.

            .. note::
                This class is designed to integrate with a base extractor and extend
                its capabilities with LLM-based proposition extraction mechanisms.

            Attributes:
                llm (BaseLLM): An instance of the base LLM to process predictions.
                prompt_template (str): The template guiding the prompt for the LLM.

            Methods:
                _extract_propositions: Extracts propositions, applying the prompt and LLM model.
            """
            return self.llm.predict(
                PromptTemplate(template=self.prompt_template), text=text
            )

        coro = asyncio.to_thread(blocking_llm_call)

        raw_response = await coro

        propositions = raw_response.split('\n')

        unique_propositions = {p: None for p in propositions if p}

        return Propositions(propositions=list(unique_propositions.keys()))
