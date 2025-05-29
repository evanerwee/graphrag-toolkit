# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List

from graphrag_toolkit.lexical_graph.storage.vector.vector_index import VectorIndex

class VectorIndexFactoryMethod():
    """
    A factory method for creating vector indexes.

    This abstract class defines an interface for creating vector indexes. Any subclass should
    implement the `try_create` method to specify creation logic. The factory method ensures a
    consistent way of generating vector indexes, potentially handling input validation,
    additional configurations, or other pre-processing tasks.

    Methods in any subclasses are expected to return valid instances of `VectorIndex` based
    on provided input.

    Attributes:
        None
    """
    @abc.abstractmethod
    def try_create(self, index_names:List[str], vector_index_info:str, **kwargs) -> List[VectorIndex]:
        raise NotImplementedError