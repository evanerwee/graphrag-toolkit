# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Union, Dict, Generator, Iterable

from llama_index.core.schema import TextNode, Document, BaseNode
from llama_index.core.schema import NodeRelationship


class SourceDocument(BaseModel):
    """
    Represents a source document comprising a collection of nodes and an optional reference
    node.

    This class is designed to store and manage a list of nodes, along with an optional
    reference node, within a strict configuration environment. It provides functionality
    to retrieve information about the source node if present.

    :ivar refNode: An optional reference node associated with the document.
    :type refNode: Optional[BaseNode]
    :ivar nodes: A list of nodes associated with the source document.
    :type nodes: List[BaseNode]
    """

    model_config = ConfigDict(strict=True)

    refNode: Optional[BaseNode] = None
    nodes: List[BaseNode] = []

    def source_id(self):
        """
        Retrieves the source ID of the first node in the `nodes` list.

        The source ID is determined by accessing the `relationships` attribute of the
        first node and returning the `node_id` associated with the
        `NodeRelationship.SOURCE` relationship type. If the `nodes` list is empty,
        this method returns `None`.

        :return: The source ID of the first node, or `None` if no nodes are present.
        :rtype: int or None
        """
        if not self.nodes:
            return None
        return self.nodes[0].relationships[NodeRelationship.SOURCE].node_id


SourceType = Union[SourceDocument, BaseNode]


def source_documents_from_source_types(
    inputs: Iterable[SourceType],
) -> Generator[SourceDocument, None, None]:
    """
    Transforms and yields SourceDocument objects from provided input data of various types.

    The function processes a sequence of inputs, which can include multiple types:
    SourceDocument, Document, and TextNode. Based on the type of each input, it
    either yields the input as-is or constructs new SourceDocument objects by
    aggregating related nodes. The function organizes nodes based on their source
    information when they are of type TextNode.

    :param inputs: An iterable collection of SourceType objects, including
        SourceDocument, Document, or TextNode.
    :return: A generator yielding instances of SourceDocument created or processed
        from the input data.
    """
    chunks_by_source: Dict[str, SourceDocument] = {}

    for i in inputs:
        if isinstance(i, SourceDocument):
            yield i
        elif isinstance(i, Document):
            yield SourceDocument(nodes=[i])
        elif isinstance(i, TextNode):
            source_info = i.relationships[NodeRelationship.SOURCE]
            source_id = source_info.node_id
            if source_id not in chunks_by_source:
                chunks_by_source[source_id] = SourceDocument()
            chunks_by_source[source_id].nodes.append(i)
        else:
            raise ValueError(f'Unexpected source type: {type(i)}')

    for nodes in chunks_by_source.values():
        yield SourceDocument(nodes=list(nodes))


class Propositions(BaseModel):
    """
    Represents a data model for handling a list of propositions.

    This class serves as a model to store and manage a collection of propositions
    and provides strict validation on its attributes via the associated configuration.

    :ivar model_config: Configuration for the model, allowing strict type validation.
    :type model_config: ConfigDict
    :ivar propositions: List of propositions represented as strings.
    :type propositions: List[str]
    """

    model_config = ConfigDict(strict=True)

    propositions: List[str]


class Entity(BaseModel):
    """
    Represents an entity model for managing data with optional strict configuration.

    This class is designed to represent an entity with specific attributes such as
    entity ID, value, and classification. It inherits from `BaseModel` and allows
    for optional strict configuration using `ConfigDict(strict=True)`. The entity ID
    and classification are optional attributes, while the value is required.

    :ivar entityId: An optional unique identifier for the entity.
    :type entityId: Optional[str]
    :ivar value: A required value representing the main data of the entity.
    :type value: str
    :ivar classification: An optional classification associated with the entity.
    :type classification: Optional[str]
    """

    model_config = ConfigDict(strict=True)

    entityId: Optional[str] = None

    value: str
    classification: Optional[str] = None


class Relation(BaseModel):
    """
    Represents a relation model with a strict configuration.

    This class is designed to represent a relation with a specific configuration. The
    configuration ensures strict validation for the model attributes. It inherits from
    BaseModel to provide model validation and management capabilities.

    :ivar value: Represents the value of the relation.
    :type value: str
    """

    model_config = ConfigDict(strict=True)

    value: str


class Fact(BaseModel):
    """
    Represents a structured fact consisting of a subject, predicate, and object,
    typically to model relationships or assertions in data. This class uses strict
    configuration to enforce data validation rules.

    :ivar factId: Identifier for the fact, which is optional.
    :type factId: Optional[str]
    :ivar statementId: Identifier for the statement associated with the fact, which
        is optional.
    :type statementId: Optional[str]
    :ivar subject: The entity representing the subject in the fact.
    :type subject: Entity
    :ivar predicate: The relationship or action linking the subject to the object.
    :type predicate: Relation
    :ivar object: The entity representing the object in the fact, which is optional.
    :type object: Optional[Entity]
    :ivar complement: Additional information or context about the fact,
        which is optional.
    :type complement: Optional[str]
    """

    model_config = ConfigDict(strict=True)

    factId: Optional[str] = None
    statementId: Optional[str] = None

    subject: Entity
    predicate: Relation
    object: Optional[Entity] = None
    complement: Optional[str] = None


class Statement(BaseModel):
    """
    Represents a statement with associated attributes such as its unique identifier,
    related topic and chunk IDs, and accompanying details and facts.

    The Statement class provides a structured way to manage and organize information
    related to a specific statement, including its value, additional details, and
    factual data. Instances of this class can be used to encapsulate and manipulate
    information relevant to a statement for a variety of use cases.

    :ivar statementId: Unique identifier of the statement.
    :type statementId: Optional[str]
    :ivar topicId: Identifier for the topic associated with the statement.
    :type topicId: Optional[str]
    :ivar chunkId: Identifier for the chunk of content the statement belongs to.
    :type chunkId: Optional[str]
    :ivar value: The textual content of the statement.
    :type value: str
    :ivar details: A list of additional details or clarifications about the statement.
    :type details: List[str]
    :ivar facts: A collection of factual information associated with the statement.
    :type facts: List[Fact]
    """

    model_config = ConfigDict(strict=True)

    statementId: Optional[str] = None
    topicId: Optional[str] = None
    chunkId: Optional[str] = None

    value: str
    details: List[str] = []
    facts: List[Fact] = []


class Topic(BaseModel):
    """
    Represents a Topic with related metadata such as `value`, `entities`,
    and `statements`, as well as identifiers including `topicId` and `chunkIds`.

    This class serves as a data model for organizing and managing topic-related
    information. It is constructed with a strict configuration to ensure valid
    data input. It binds additional metadata such as associated entities and
    statements for comprehensive information tracking.

    :ivar topicId: The unique identifier for the topic (if available). Defaults to None.
    :type topicId: Optional[str]
    :ivar chunkIds: A list of chunk identifiers associated with the topic. Defaults to an empty list.
    :type chunkIds: List[str]
    :ivar value: The main textual value or description of the topic.
    :type value: str
    :ivar entities: A list of entities related to the topic. Defaults to an empty list.
    :type entities: List[Entity]
    :ivar statements: A list of statements associated with the topic. Defaults to an empty list.
    :type statements: List[Statement]
    """

    model_config = ConfigDict(strict=True)

    topicId: Optional[str] = None
    chunkIds: List[str] = []

    value: str
    entities: List[Entity] = []
    statements: List[Statement] = []


class TopicCollection(BaseModel):
    """
    Represents a collection of topics.

    This class is designed to manage and store multiple topics efficiently. It
    inherits from the BaseModel class and provides strict configuration and data
    validation. The primary purpose of this class is to hold a list of topics,
    which can be easily accessed and utilized in applications where grouped topics
    need to be managed.

    :ivar model_config: Configuration settings for the model, enforcing strict
        validation rules and behavior.
    :type model_config: ConfigDict
    :ivar topics: A list of topics managed by this collection.
    :type topics: List[Topic]
    """

    model_config = ConfigDict(strict=True)

    topics: List[Topic] = []
