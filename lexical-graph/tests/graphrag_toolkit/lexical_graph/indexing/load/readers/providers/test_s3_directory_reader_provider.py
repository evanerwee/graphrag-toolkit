# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

def test_raises_exception_if_dependencies_not_installed():
    from graphrag_toolkit.lexical_graph.indexing.load.readers.providers import S3DirectoryReaderProvider
    from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_config import S3DirectoryReaderConfig 
       
    with pytest.raises(ImportError) as exc_info:  
         reader = S3DirectoryReaderProvider(S3DirectoryReaderConfig(key='my-key'))
 
    assert exc_info.value.args[0] == "llama-index-readers-s3 package not found, install with 'pip install llama-index-readers-s3'"