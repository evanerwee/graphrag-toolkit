# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

def test_raises_exception_if_dependencies_not_installed():
    from graphrag_toolkit.lexical_graph.indexing.load.readers.providers import WikipediaReaderProvider
    from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import WikipediaReaderConfig 
       
    with pytest.raises(ImportError) as exc_info:  
         reader = WikipediaReaderProvider(WikipediaReaderConfig())
 
    assert exc_info.value.args[0] == "llama-index-readers-wikipedia package not found, install with 'pip install llama-index-readers-wikipedia'"