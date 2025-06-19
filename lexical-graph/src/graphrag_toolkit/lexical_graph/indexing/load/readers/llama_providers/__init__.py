# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LlamaIndex-based document extraction providers.

This package contains extraction providers that wrap LlamaIndex document readers
to provide a consistent interface for extracting documents from various sources.
"""

from graphrag_toolkit.lexical_graph.indexing.load.readers.reader_provider_registry import ReaderProviderRegistry

from .directory_reader_provider import DirectoryReaderProvider
from .docx_reader_provider import DocxReaderProvider
from .github_repo_provider import GitHubReaderProvider
from .pdf_reader_provider import PDFReaderProvider
from .pptx_reader_provider import PPTXReaderProvider
from .s3_directory_reader_provider import S3DirectoryReaderProvider
from .web_reader_provider import WebReaderProvider
from .youtube_reader_provider import YouTubeReaderProvider

ReaderProviderRegistry.register("directory", DirectoryReaderProvider)
ReaderProviderRegistry.register("docx", DocxReaderProvider)
ReaderProviderRegistry.register("github", GitHubReaderProvider)
ReaderProviderRegistry.register("pdf", PDFReaderProvider)
ReaderProviderRegistry.register("pptx", PPTXReaderProvider)
ReaderProviderRegistry.register("s3_directory", S3DirectoryReaderProvider)
ReaderProviderRegistry.register("web", WebReaderProvider)
ReaderProviderRegistry.register("youtube", YouTubeReaderProvider)
