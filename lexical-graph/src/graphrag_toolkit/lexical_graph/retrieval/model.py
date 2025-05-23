# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field, AliasChoices
from typing import List, Optional, Union, Dict


class Statement(BaseModel):
    """Represents a statement model with associated attributes for processing
    statements, their details, facts, and related metadata.

    The `Statement` class is intended to model a structured statement object, allowing the
    definition of a main statement, optional metadata such as statement ID, related facts,
    chunk reference, calculated score, and an alternative string representation of the
    statement. This model ensures strict validation of its attributes and enforces data
    consistency.

    Attributes:
        statementId (Optional[str]): Optional unique identifier of the statement.
        statement (str): The main content of the statement.
        facts (List[str]): A list of related factual statements.
        details (Optional[str]): Additional details or explanations for the statement.
        chunkId (Optional[str]): Identifier for the chunk to which this statement belongs.
        score (Optional[float]): Optional score or relevance metric for the statement.
        statement_str (Optional[str]): Optional alternative string representation of the statement.
    """

    model_config = ConfigDict(strict=True)

    statementId: Optional[str] = None
    statement: str
    facts: List[str] = []
    details: Optional[str] = None
    chunkId: Optional[str] = None
    score: Optional[float] = None
    statement_str: Optional[str] = None


StatementType = Union[Statement, str]


class Chunk(BaseModel):
    """Represents a Chunk model with strict configuration.

    This class defines a chunk structure with attributes such as its unique
    identifier, an optional value, and an optional score. It is primarily used
    to store and manage data related to individual chunks within a dataset or
    system. The class ensures strict validation of its structure via the
    inherited `ConfigDict` strict mode.

    Attributes:
        chunkId (str): A unique identifier for the chunk.
        value (Optional[str]): An optional value associated with the chunk.
        score (Optional[float]): An optional score indicating some metric
        related to the chunk.
    """

    model_config = ConfigDict(strict=True)

    chunkId: str
    value: Optional[str] = None
    score: Optional[float] = None


class Topic(BaseModel):
    """Represents a topic containing related chunks and statements.

    This class defines a topic with associated chunks and statements. It can
    be used to organize data relevant to a particular subject, ensuring
    relationships and groupings are maintained for further processing or
    analysis.

    Attributes:
        model_config (ConfigDict): Configuration settings for model behavior, enforcing strict type checks.
        topic (str): The main subject of the topic.
        chunks (List[Chunk]): A collection of associated chunks related to the topic.
        statements (List[StatementType]): Statements or assertions associated with the topic.
    """

    model_config = ConfigDict(strict=True)

    topic: str
    chunks: List[Chunk] = []
    statements: List[StatementType] = []


class Source(BaseModel):
    """Represents a source entity with a unique identifier and associated
    metadata.

    This class is used to encapsulate information about a data source, including a
    unique identifier (sourceId) and optional metadata as key-value pairs. It ensures
    that data adheres to a strict schema through its configuration.

    Attributes:
        sourceId (str): A unique identifier for the source.
        metadata (Dict[str, str]): A dictionary representing additional information about the source, where keys and values are both strings. Defaults to an empty dictionary.
    """

    model_config = ConfigDict(strict=True)

    sourceId: str
    metadata: Dict[str, str] = {}


SourceType = Union[str, Source]


class SearchResult(BaseModel):
    """Represents the result of a search operation.

    This class models the result of a search operation, including the source of
    the result, relevant topics, selected topic, associated statements, and an
    optional score indicating the quality or relevance of the result.

    Attributes:
        source (SourceType): The source from which the search result was obtained.
        topics (List[Topic]): A list of topics relevant to the search result.
        topic (Optional[str]): An optional single selected topic associated with the search result.
        statements (List[StatementType]): A list of statements relevant to the search result.
        score (Optional[float]): An optional score indicating the quality or relevance of the search result.
    """

    model_config = ConfigDict(strict=True)

    source: SourceType
    topics: List[Topic] = []
    topic: Optional[str] = None
    statements: List[StatementType] = []
    score: Optional[float] = None


class Entity(BaseModel):
    """Represents an entity model with specific configurations and attributes.

    This class defines an entity structure with attributes for ID, value, and
    classification. It enforces strict configuration settings through a model
    configuration and supports aliasing for specific fields. Typically used to
    represent structured data objects with validation in applications requiring
    strict field type adherence.

    Attributes:
        entityId (str): Unique identifier for the entity.
        value (str): Value or content associated with the entity.
        classification (str): Category or classification of the entity, supporting alias
        fields such as 'class' or 'classification'.
    """

    model_config = ConfigDict(strict=True)

    entityId: str
    value: str
    classification: str = Field(alias=AliasChoices('class', 'classification'))


class ScoredEntity(BaseModel):
    """Represents an entity with an associated score.

    This class extends the BaseModel and is used to link an Entity object
    with a corresponding score, ensuring strict validation of its properties.

    Attributes:
        entity (Entity): The entity object associated with the score.
        score (float): The score value associated with the entity.
    """

    model_config = ConfigDict(strict=True)

    entity: Entity
    score: float


class SearchResultCollection(BaseModel):
    """Represents a collection of search results and associated scored
    entities.

    The class is designed to store and manage a list of search results and
    a list of scored entities resulting from a search operation. It provides
    methods to add search results and entities, as well as to replace
    the current set of search results with a new set.

    Attributes:
        model_config: Configuration settings for the model. The strict setting
        ensures that data types and structures are strictly enforced.
        results (List[SearchResult]): A list of search result objects.
        entities (List[ScoredEntity]): A list of scored entities associated
        with the search results.
    """

    model_config = ConfigDict(strict=True)

    results: List[SearchResult] = []
    entities: List[ScoredEntity] = []

    def add_search_result(self, result: SearchResult):
        """Adds a search result to the existing list of results.

        This method appends the given search result to the internal list of
        results maintained by the instance. It is intended to manage the
        storage of search results during execution, ensuring they are added
        in the order they are received.

        Args:
            result: A SearchResult object representing the search result
            to be added to the list of results.
        """
        self.results.append(result)

    def add_entity(self, entity: ScoredEntity):
        """Adds an entity to the list of entities, either merging it with an
        existing entity or appending it as a new entry.

        If the entity already exists in the list (based on a matching `entityId`),
        its score is incremented by the score of the added entity. Otherwise, the
        new entity is appended to the list.

        Args:
            entity (ScoredEntity): The scored entity to add or merge with an existing entity.
        """
        if self.entities is None:
            self.entities = []
        existing_entity = next(
            (x for x in self.entities if x.entity.entityId == entity.entity), None
        )
        if existing_entity:
            existing_entity.score += entity.score
        else:
            self.entities.append(entity)

    def with_new_results(self, results: List[SearchResult]):
        """Updates the current object with new search results and returns the
        updated object. This allows for fluent interface pattern by returning
        the instance itself after updating its state with the provided search
        results.

        Args:
            results (List[SearchResult]): A list of SearchResult objects to update the instance's current
            results data.

        Returns:
            The current instance with the updated results.
        """
        self.results = results
        return self
