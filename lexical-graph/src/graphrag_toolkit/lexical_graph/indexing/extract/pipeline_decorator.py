# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import six
from typing import Iterable

from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument

@six.add_metaclass(abc.ABCMeta)
class PipelineDecorator():
    """
    Abstract base class for defining pipeline decorators.

    This class provides an interface for creating pipeline decorators that
    process input documents and transform them through an arbitrary operation.
    It is intended to be subclassed, with the abstract methods implemented
    to define custom behavior for handling documents in a pipeline.

    Attributes:
        None
    """
    @abc.abstractmethod
    def handle_input_docs(self, docs:Iterable[SourceDocument]) -> Iterable[SourceDocument]:
        """
        Abstract method that processes a collection of SourceDocument instances
        and returns an iterable of processed SourceDocument instances.

        This method defines an interface for handling input documents that must
        be implemented by subclasses. The implementation of the method should
        provide the logic for processing the documents within the iterable input.

        Args:
            docs (Iterable[SourceDocument]): An iterable collection of SourceDocument
                objects to be processed.

        Returns:
            Iterable[SourceDocument]: An iterable collection of processed SourceDocument
                objects after applying the logic defined in the subclass implementation.
        """
        pass

    @abc.abstractmethod
    def handle_output_doc(self, doc: SourceDocument) -> SourceDocument:
        """
        An abstract method to process and handle an input SourceDocument object
        and return the processed SourceDocument.

        Args:
            doc (SourceDocument): The input document that needs to be processed.

        Returns:
            SourceDocument: The processed document after handling.
        """
        pass

