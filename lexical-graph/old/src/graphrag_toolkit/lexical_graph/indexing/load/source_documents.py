# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List
from llama_index.core import Document


class SourceDocuments:
    """
    Represents a collection of source document providers.

    The class is designed to manage and iterate over collections of documents provided
    by callable objects. Each callable in the `source_documents_fns` attribute is expected
    to return an iterable containing document data. The class handles the complexities
    of nested iterable structures, providing a unified mechanism to retrieve individual
    document items. This enables streamlined access to document data across various
    sources.

    :ivar source_documents_fns: A list of callables that each return a collection of document
        data when executed.
    :type source_documents_fns: List[Callable[[], List[Document]]]
    """

    def __init__(self, source_documents_fns: List[Callable[[], List[Document]]]):
        """
        Initializes an instance with a list of callable functions, each returning
        a list of documents when invoked. This class constructor is intended to
        store a collection of functions for fetching or generating documents.

        :param source_documents_fns: A list of callable functions, where each function
            returns a list of Document objects when invoked.
        :type source_documents_fns: List[Callable[[], List[Document]]]
        """
        self.source_documents_fns = source_documents_fns

    def __iter__(self):
        """
        Iterates through all documents derived from the provided source document functions.

        This method recursively iterates through a hierarchy of nested lists contained
        in the source document function outputs, yielding each individual element.

        :return: A generator that yields each document or individual element
                 derived from the source documents.
        :rtype: Iterator[Any]
        """
        for source_documents_fn in self.source_documents_fns:
            for source_documents in source_documents_fn():
                if isinstance(source_documents, list):
                    for item in source_documents:
                        if isinstance(item, list):
                            for i in item:
                                yield i
                        else:
                            yield item
                else:
                    yield source_documents
