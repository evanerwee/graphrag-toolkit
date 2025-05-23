# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import os
import io
import uuid
import logging
import shutil
import copy
import base64
from pathlib import Path
from typing import Callable, Dict, Any
from os.path import join
from urllib.parse import urlparse

from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.load.file_based_chunks import (
    FileBasedChunks,
)
from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument
from graphrag_toolkit.lexical_graph.indexing.extract.id_rewriter import IdRewriter
from graphrag_toolkit.lexical_graph import GraphRAGConfig

from llama_index.core.schema import TextNode, Document
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

logger = logging.getLogger(__name__)


class TempFile:
    """
    Represents a file handler object used for managing and processing file-related operations.

    This class provides the foundation for storing the file path and allows future extensions
    for interacting with the file contents or metadata.

    :ivar filepath: The path to the file that this object relates to.
    :type filepath: str
    :ivar file: The file object representing the opened file during the context manager usage.
    :type file: file object
    """

    def __init__(self, filepath):
        """
        Represents a class that manages or processes a given file path.

        This class initializes with a file path, which is essential for further
        functionality related to file manipulation, reading, writing, or other
        operations.

        :param filepath: The path to a file that this class will handle or manage.
        :type filepath: str
        """
        self.filepath = filepath

    def __enter__(self):
        """
        Represents the entry point for a context manager that opens a file.

        This method is specifically designed to handle the setup process when
        using the `with` statement on an instance of the class. It ensures the
        internal file is opened successfully and returns the current instance
        of the context manager object.

        :return: The instance of the context manager for use within the `with`
                 block.
        :rtype: object
        """
        self.file = open(self.filepath)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Cleans up resources when exiting a context managed block by closing the associated
        file and removing its file from the file system.

        This method is called when the context manager is exited. It ensures that any
        associated file is properly closed, and its corresponding file on the filesystem
        is deleted as part of cleanup.

        :param exception_type: The class of the exception raised in the context, if any.
            If no exception was raised, this will be None.
        :param exception_value: The instance of the exception raised in the context,
            if any. If no exception was raised, this will be None.
        :param exception_traceback: The traceback object related to the exception raised
            in the context, if any. If no exception was raised, this will be None.
        :return: None
        """
        self.file.close()
        os.remove(self.filepath)

    def readline(self):
        """
        Reads a single line from the associated file object.

        This method reads the next line from the file represented by the `file`
        attribute of the instance. The line will include the trailing newline
        character, if it exists. If the end of the file is reached, an empty
        string will be returned.

        :return: The next line from the file as a string, including the newline
            character, or an empty string if the end of the file is reached.
        :rtype: str
        """
        return self.file.readline()


class TempDir:
    """
    Represents a temporary directory context manager.

    This class provides a context-managed approach to ensure the existence
    of a directory during the lifetime of a context and then cleans it up
    upon exiting the context. It is useful for managing temporary directories
    in a clean and automated manner.

    :ivar dir_path: A string representing the directory path that this context
        manager will manage.
    :type dir_path: str
    """

    def __init__(self, dir_path):
        """
        Represents a class that manages a directory path.

        This class provides an interface to hold and represent a directory
        path. Users can initialize an instance of this class with a specific
        directory path.

        :param dir_path: The path to the directory being managed (as a string)
        :type dir_path: str
        """
        self.dir_path = dir_path

    def __enter__(self):
        """
        Handles the initialization of a context manager for ensuring the
        existence of a specific directory. When entering the context, if
        the directory does not exist, it is created. This ensures the
        directory is available for further operations during the context
        manager's lifecycle.

        :return: The context manager instance
        :rtype: Self
        """
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Exit the runtime context and remove the directory at the specified path if it exists.

        This method is typically used in combination with the context management protocol. Upon
        exiting the context, the method ensures the directory specified by the object's dir_path
        attribute is removed. If the directory does not exist, no action is taken.

        :param exception_type: Type of the exception raised (if any) during the execution of the
            code block within the context.
        :type exception_type: type[BaseException] | None
        :param exception_value: The exception instance raised (if any) during the execution of
            the code block within the context.
        :type exception_value: BaseException | None
        :param exception_traceback: Traceback object associated with the raised exception (if any).
        :type exception_traceback: traceback | None
        :return: Always returns False to propagate the exception, allowing it to be handled
            outside the context manager if raised.
        :rtype: bool
        """
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)


class BedrockKnowledgeBaseExport:
    """
    Handles the S3 export process, enabling the downloading, parsing, and management
    of data from an Amazon Bedrock Knowledge Base. The class provides functionalities
    to process and save data chunks and source documents in a structured local directory.

    It is equipped with configurations to support selective operations, such as filtering
    by S3 key prefix, including additional metadata, and optional embedding integration.

    :ivar bucket_name: Name of the S3 bucket containing the Knowledge Base data.
    :type bucket_name: str
    :ivar key_prefix: Prefix to filter the S3 objects in the specified bucket.
    :type key_prefix: str
    :ivar region: AWS region containing the S3 bucket.
    :type region: str
    :ivar limit: Maximum number of S3 objects to process (-1 for no limit).
    :type limit: int
    :ivar output_dir: Directory path where output files will be written.
    :type output_dir: str
    :ivar s3_client: AWS S3 client used for interacting with the S3 service.
    :type s3_client: object
    :ivar id_rewriter: Facilitates rewriting unique IDs with a custom tenant identifier.
    :type id_rewriter: object
    :ivar metadata_fn: Optional function to produce or modify object metadata.
    :type metadata_fn: Callable[[str], Dict[str, Any]]
    :ivar include_embeddings: Specifies whether embeddings should be included.
    :type include_embeddings: bool
    :ivar include_source_doc: Specifies whether to include source documents in processing.
    :type include_source_doc: bool
    """

    def __init__(
        self,
        region: str,
        bucket_name: str,
        key_prefix: str,
        limit: int = -1,
        output_dir: str = 'output',
        metadata_fn: Callable[[str], Dict[str, Any]] = None,
        include_embeddings: bool = True,
        include_source_doc: bool = False,
        tenant_id: str = None,
        **kwargs,
    ):
        """
        Initializes an instance of the class, setting up configuration
        parameters such as the AWS S3 bucket details, file processing
        configurations, and optional metadata handling.

        :param region: The AWS region for the S3 bucket.
        :param bucket_name: The name of the AWS S3 bucket.
        :param key_prefix: The prefix path in the S3 bucket from where data
            will be accessed.
        :param limit: The maximum number of objects to process from the S3
            bucket. Defaults to -1, which implies no limit.
        :param output_dir: The directory to which the output files will be
            saved locally.
        :param metadata_fn: A callable function that accepts a string and
            returns a dictionary containing metadata for processing files.
        :param include_embeddings: Determines whether to include embeddings
            in the processing. Defaults to True.
        :param include_source_doc: Specifies whether to include source
            documents during processing. Defaults to False.
        :param tenant_id: Identifier for the tenant for managing tenant-specific
            configurations. Can be None.
        :param kwargs: Additional keyword arguments for custom processing
            configurations.
        """
        self.bucket_name = bucket_name
        self.key_prefix = key_prefix
        self.region = region
        self.limit = limit
        self.output_dir = output_dir
        self.s3_client = GraphRAGConfig.s3
        self.id_rewriter = IdRewriter(id_generator=IdGenerator(tenant_id=tenant_id))
        self.metadata_fn = metadata_fn
        self.include_embeddings = include_embeddings
        self.include_source_doc = include_source_doc

    def _kb_chunks(self, kb_export_dir):
        """
        Fetches and processes chunks of data from knowledge base export files stored in an S3 bucket.

        This function uses the S3 client to retrieve paginated results for objects stored in the specified
        S3 bucket and with the specified key prefix. For each object, it downloads the file temporarily
        to the provided knowledge base export directory and reads its content line by line. Each line
        is assumed to be a JSON object, which is then parsed and yielded.

        :param kb_export_dir: The directory where temporary files for processing the knowledge base
            exports should be stored.
        :type kb_export_dir: str

        :return: Yields parsed JSON objects from the knowledge base export files stored in S3.
        :rtype: Iterator[dict]
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.key_prefix)

        keys = [obj['Key'] for page in pages for obj in page['Contents']]

        for key in keys:

            logger.info(
                f'Loading Amazon Bedrock Knowledge Base export file [bucket: {self.bucket_name}, key: {key}, region: {self.region}]'
            )

            temp_filepath = join(kb_export_dir, f'{uuid.uuid4().hex}.json')
            self.s3_client.download_file(self.bucket_name, key, temp_filepath)

            with TempFile(temp_filepath) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    else:
                        yield json.loads(line)

    def _parse_key(self, source):
        """
        Parses the given source URL and extracts the path component,
        excluding any leading slashes. This function utilizes the `urlparse`
        library to parse the input and ensure fragments are ignored during
        processing.

        :param source: The source URL string to be parsed.
        :type source: str
        :return: The extracted path component from the URL, with leading
            slashes removed.
        :rtype: str
        """
        parsed = urlparse(source, allow_fragments=False)
        return parsed.path.lstrip('/')

    def _download_source_doc(self, source, doc_file_path):
        """
        Downloads the source document from an Amazon S3 bucket and processes it to generate
        a formatted document. It includes metadata extraction, content decoding, and saving
        the final document to a specified file path.

        :param source: The identifier for the document's source.
        :type source: str
        :param doc_file_path: The path where the processed document will be saved.
        :type doc_file_path: str
        :return: A processed instance of the Document object containing the downloaded data
            and associated metadata.
        :rtype: Document
        """
        key = self._parse_key(source)

        logger.debug(
            f'Loading Amazon Bedrock Knowledge Base underlying source document [source: {source}, bucket: {self.bucket_name}, key: {key}, region: {self.region}]'
        )

        object_metadata = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
        content_type = object_metadata.get('ContentType', None)

        with io.BytesIO() as io_stream:
            self.s3_client.download_fileobj(self.bucket_name, key, io_stream)

            io_stream.seek(0)

            if content_type and content_type in ['application/pdf']:
                data = base64.b64encode(io_stream.read())
            else:
                data = io_stream.read().decode('utf-8')

        metadata = self.metadata_fn(data) if self.metadata_fn else {}

        if 'source' not in metadata:
            metadata['source'] = source

        doc = Document(text=data, metadata=metadata)

        doc = self.id_rewriter([doc])[0]

        with open(doc_file_path, 'w') as f:
            f.write(doc.to_json())

        return doc

    def _open_source_doc(self, doc_file_path):
        """
        Opens a source document file, reads its content, deserializes it from JSON,
        and converts it into a `Document` object using the `Document.from_dict` method.

        This function is designed to facilitate the process of loading a document
        stored in JSON format into a structured `Document` object.

        :param doc_file_path: The file path to the source document in JSON format.
                              It must be a valid path to a JSON file containing the
                              serialized document data.
        :type doc_file_path: str
        :return: A `Document` object deserialized from the given JSON file.
        :rtype: Document
        """
        with open(doc_file_path) as f:
            data = json.load(f)
            return Document.from_dict(data)

    def _get_source_doc(self, source_docs_dir, source):
        """
        Retrieve or download the source document associated with the given source
        identifier. If the document already exists in the specified directory, it is
        opened and returned. If not, the necessary directories are created, and the
        document is downloaded and stored.

        :param source_docs_dir: The directory where source documents are stored.
            This directory should contain subdirectories corresponding to unique
            hashed IDs for each source.
        :type source_docs_dir: str
        :param source: The source identifier from which the unique hash is generated.
            This ID is used to locate or store the corresponding document.
        :type source: str
        :return: Returns the content of the source document either from the local
            storage or after downloading it to the specified directory.
        :rtype: str
        """
        source_id = get_hash(source)
        doc_directory_path = join(source_docs_dir, source_id, 'document')
        doc_file_path = join(doc_directory_path, 'source_doc')

        if os.path.exists(doc_file_path):
            return self._open_source_doc(doc_file_path)
        else:
            if not os.path.exists(doc_directory_path):
                os.makedirs(doc_directory_path)
            return self._download_source_doc(source, doc_file_path)

    def _save_chunk(self, source_docs_dir, chunk, source):
        """
        Saves a specific chunk of data to the appropriate directory based on the source.
        This involves creating a directory structure using the source identifier and chunk
        identifier, then writing the JSON representation of the chunk to a file. If the
        directory path does not already exist, it will be created.

        :param source_docs_dir: Path to the directory where source documents are stored
        :type source_docs_dir: str
        :param chunk: The chunk of data to be saved
        :type chunk: Chunk
        :param source: The original source data used to create the chunk
        :type source: str
        :return: None
        """
        chunk = self.id_rewriter([chunk])[0]

        source_id = get_hash(source)
        chunks_directory_path = join(source_docs_dir, source_id, 'chunks')
        chunk_file_path = join(chunks_directory_path, chunk.id_)

        if not os.path.exists(chunks_directory_path):
            os.makedirs(chunks_directory_path)

        with open(chunk_file_path, 'w') as f:
            f.write(chunk.to_json())

    def _get_doc_count(self, source_docs_dir):
        """
        Calculate the total document count in the given source directory, excluding one file.

        :param source_docs_dir: Path to the directory containing the source documents
            as a string.
        :type source_docs_dir: str
        :return: The total count of documents in the directory subtracted by one.
        :rtype: int
        """
        doc_count = (
            len([name for name in os.listdir(source_docs_dir) if os.path.isfile(name)])
            - 1
        )
        logger.info(f'doc_count: {doc_count}')
        return doc_count

    def docs(self):
        """
        Represents a method to return the current instance of the class. This method might
        be used for purposes such as method chaining or simply to access the current object.

        :return: The instance of the current class
        :rtype: self
        """
        return self

    def _with_page_number(self, metadata, page_number):
        """
        Creates a deep copy of the provided metadata dictionary and updates it with the
        given page number if it is provided. If the page number is not provided, returns
        the original metadata dictionary unmodified.

        :param metadata: The original metadata dictionary to be copied and potentially
            updated.
        :type metadata: dict
        :param page_number: The page number to be added to the metadata copy. If not
            provided, the metadata remains unchanged.
        :type page_number: int or None
        :return: A new dictionary with the page number included if provided; otherwise,
            the original metadata.
        :rtype: dict
        """
        if page_number:
            metadata_copy = copy.deepcopy(metadata)
            metadata_copy['page_number'] = page_number
            return metadata_copy
        else:
            return metadata

    def __iter__(self):
        """
        Creates a generator function that iteratively processes knowledge base chunks and
        yields SourceDocument objects generated from these chunks.

        The function handles temporary directory creation, manages various chunks of data,
        parses chunk metadata, associates it with its source documents, and optionally includes
        embeddings if enabled. The function iterates over the processed data and yields
        the resulting source documents along with their associated nodes.

        :param self: Instance of the class containing the method call.
        :return: A generator yielding `SourceDocument` objects that contain processed
                 metadata and associated nodes.
        """
        job_dir = join(self.output_dir, 'bedrock-kb-export', f'{uuid.uuid4().hex}')

        bedrock_dir = join(job_dir, 'bedrock')
        llama_index_dir = join(job_dir, 'llama-index')

        logger.info(
            f'Creating Amazon Bedrock Knowledge Base temp directories [bedrock_dir: {bedrock_dir}, llama_index_dir: {llama_index_dir}]'
        )

        count = 0

        with TempDir(job_dir) as j, TempDir(bedrock_dir) as k, TempDir(
            llama_index_dir
        ) as s:

            for kb_chunk in self._kb_chunks(bedrock_dir):

                bedrock_id = kb_chunk['id']
                page_number = kb_chunk.get(
                    'x-amz-bedrock-kb-document-page-number', None
                )
                metadata = json.loads(kb_chunk['AMAZON_BEDROCK_METADATA'])
                source = metadata['source']

                source_doc = self._get_source_doc(llama_index_dir, source)

                chunk = TextNode()

                chunk.text = kb_chunk['AMAZON_BEDROCK_TEXT']
                chunk.metadata = metadata
                chunk.metadata['bedrock_id'] = bedrock_id
                if self.include_embeddings:
                    chunk.embedding = kb_chunk['bedrock-knowledge-base-default-vector']
                chunk.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=source_doc.id_,
                    node_type=NodeRelationship.SOURCE,
                    metadata=source_doc.metadata,
                    hash=source_doc.hash,
                )

                self._save_chunk(llama_index_dir, chunk, source)

            for d in [d for d in Path(llama_index_dir).iterdir() if d.is_dir()]:

                document = None

                if self.include_source_doc:
                    source_doc_file_path = join(d, 'document', 'source_doc')
                    with open(source_doc_file_path) as f:
                        document = Document.from_json(f.read())

                file_based_chunks = FileBasedChunks(str(d), 'chunks')
                chunks = [c for c in file_based_chunks.chunks()]

                yield SourceDocument(refNode=document, nodes=chunks)

                count += 1
                if self.limit > 0 and count >= self.limit:
                    break
