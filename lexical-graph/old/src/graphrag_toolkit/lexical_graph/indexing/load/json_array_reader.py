# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict, List, Optional, Protocol, Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class TextExtractorFunction(Protocol):
    """
    Represents a callable interface for processing input data and providing a string
    output based on the implemented processing logic.

    This class is typically used to define a protocol for functions or callable objects
    that take a dictionary of input data and return a string after performing some form
    of processing or transformation.

    :ivar __call__: Callable operation for processing input data into a string output.
    :type __call__: Callable[[Dict[str, Any]], str]
    """

    def __call__(self, data: Dict[str, Any]) -> str:
        """
        Processes the input data and returns a string result. This callable method
        is designed to take a dictionary of data as input and provide a processed
        string as output. It can be used directly on an instance of the class
        by invoking it as a function call.

        :param data: A dictionary containing keys and values, where each key is
            a string and each value can be of any type.
        :type data: Dict[str, Any]
        :return: A processed string result derived from the input data.
        :rtype: str
        """
        pass


class MetadataExtractorFunction(Protocol):
    """
    Defines a protocol for a callable object used for processing metadata.

    Instances of this protocol are intended to be invoked with a dictionary
    as input, perform some operations, and return a processed dictionary.
    This protocol serves as a blueprint for ensuring specific call-signatures
    are defined in concrete implementations.
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes an instance of the class as a callable to process input
        dictionary data and return a processed dictionary.

        Args:
            data: A dictionary containing keys and values to be processed.

        Returns:
            A dictionary containing the processed output from the input data.
        """
        pass


class JSONArrayReader(BaseReader):
    """
    Provides functionality to load, process, and transform JSON data into a structured
    format for document handling. This reader supports configuration for text extraction,
    metadata processing, and ASCII encoding.

    Designed for scenarios where JSON files need to be parsed and converted into a more
    structured format for further analysis or processing. It supports handling both
    single JSON objects and arrays of JSON objects.

    :ivar ensure_ascii: If True, non-ASCII characters are escaped in the JSON output.
    :type ensure_ascii: bool
    :ivar text_fn: Callable function for extracting text content from JSON data, allowing
                   customization or override of default behavior.
    :type text_fn: Optional[TextExtractorFunction]
    :ivar metadata_fn: Callable function for extracting metadata information, allowing
                       customization or override of default behavior.
    :type metadata_fn: Optional[MetadataExtractorFunction]
    """

    def __init__(
        self,
        ensure_ascii: bool = False,
        text_fn: Optional[TextExtractorFunction] = None,
        metadata_fn=Optional[MetadataExtractorFunction],
    ):
        """
        Initialize a new instance of the class.

        This constructor allows setting up the instance with the specified
        parameters, including optional flags and function references
        to manage the extraction behaviors.

        :param ensure_ascii: Determines whether to ensure all characters
            in the output are ASCII. Defaults to False.
        :param text_fn: A callable function reference for extracting
            text content. Defaults to None.
        :param metadata_fn: A callable function reference for extracting
            metadata content. Defaults to None.
        """
        super().__init__()
        self.ensure_ascii = ensure_ascii
        self.text_fn = text_fn
        self.metadata_fn = metadata_fn

    def _get_metadata(self, data: Dict, extra_info: Dict):
        """
        Generates metadata by combining the provided data and extra information. If an additional metadata
        generation function is specified, it is applied to the input data to further extend the metadata.

        :param data: Data from which metadata is partially derived.
        :type data: Dict
        :param extra_info: Supplementary information to include in the metadata.
        :type extra_info: Dict
        :return: Combined metadata constructed from extra information, data, and the result of the
                 metadata function if applicable.
        :rtype: Dict
        """
        metadata = {}

        if extra_info:
            metadata.update(extra_info)
        if self.metadata_fn:
            metadata.update(self.metadata_fn(data))
        return metadata

    def load_data(
        self, input_file: str, extra_info: Optional[Dict] = {}
    ) -> List[Document]:
        """
        Loads data from a file, processes it depending on the provided functions,
        and returns a list of `Document` objects. The text for each `Document`
        can be extracted using a custom function or serialized as JSON. Additional
        metadata can also be passed through `extra_info`.

        :param input_file: The file path from which data will be loaded.
        :type input_file: str
        :param extra_info: Additional information to include in the metadata.
        :type extra_info: Optional[Dict]
        :return: A list of `Document` objects containing the processed data.
        :rtype: List[Document]
        """
        with open(input_file, encoding='utf-8') as f:
            json_data = json.load(f)

            if not isinstance(json_data, list):
                json_data = [json_data]

            documents = []

            for data in json_data:
                if self.text_fn:
                    text = self.text_fn(data)
                    metadata = self._get_metadata(data, extra_info)
                    documents.append(Document(text=text, metadata=metadata))
                else:
                    json_output = json.dumps(data, ensure_ascii=self.ensure_ascii)
                    documents.append(
                        Document(
                            text=json_output,
                            metadata=self._get_metadata(data, extra_info),
                        )
                    )

            return documents
