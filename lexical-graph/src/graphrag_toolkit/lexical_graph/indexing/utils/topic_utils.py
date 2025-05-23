# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import logging
from typing import Tuple, List

from graphrag_toolkit.lexical_graph.indexing.constants import DEFAULT_TOPIC
from graphrag_toolkit.lexical_graph.indexing.model import (
    TopicCollection,
    Topic,
    Fact,
    Entity,
    Relation,
    Statement,
)

logger = logging.getLogger(__name__)


def format_text(text):
    """
    Formats the input text into a single string.

    This function takes an input, which can either be a list of strings or
    a single string. If the input is a list of strings, it joins all elements
    in the list with a newline character (`\n`) and returns the resulting
    string. If the input is already a single string, it returns the string
    unchanged.

    :param text: List of strings or a single string to be formatted.
    :type text: list[str] or str
    :return: A single formatted string. If the input is a list, it returns
        the list elements joined by newlines. If the input is a single
        string, it returns the string itself.
    :rtype: str
    """
    if isinstance(text, list):
        return '\n'.join(s for s in text)
    else:
        return text


def format_list(values: List[str]):
    """
    Formats a list of strings into a single string with each element prefixed by a dash
    and indented with spaces. Each entry in the list will appear on a new line.

    :param values: The list of strings to be formatted.
    :type values: List[str]
    :return: A string with each element in the list formatted as an indented list item.
    :rtype: str
    """
    return '\n'.join([f'   - {value}' for value in values])


def clean(s):
    """
    Cleans a given string by formatting its value and removing any surrounding
    parentheses.

    :param s: The string to be cleaned.
    :type s: str
    :return: A string that has been formatted and stripped of parentheses.
    :rtype: str
    """
    return strip_parentheses(format_value(s))


def format_value(s):
    """
    Formats the input string by replacing underscores with spaces. If the input is
    None or an empty string, it returns the input as it is.

    :param s: The input string to be formatted
    :type s: str or None
    :return: The formatted string with underscores replaced by spaces, or the
        input itself if it is None or empty
    :rtype: str
    """
    return s.replace('_', ' ') if s else ''


def format_classification(s):
    """
    Formats a given string to title case.

    This function takes a string and converts it to title case using the string
    method `.title()`. If the input string is empty or None, it returns an empty
    string.

    :param s: The input string to be formatted. It can be None.
    :type s: str or None
    :return: The formatted string in title case or an empty string if the input
             is None or empty.
    :rtype: str
    """
    return s.title() if s else ''


def strip_full_stop(s):
    """
    Removes the trailing full stop (period) from a string, if it exists. If the
    input string is empty or does not end with a full stop, the string is returned
    as-is without modifications.

    :param s: The input string that may or may not end with a full stop.
    :type s: str
    :return: The modified string without the trailing full stop, or the original
        string if no modifications were required.
    :rtype: str
    """
    return s[:-1] if s and s.endswith('.') else s


def strip_parentheses(s):
    """
    Removes text enclosed in parentheses along with the parentheses themselves
    from the given string. Extra spaces resulting from this removal are reduced,
    and leading/trailing spaces are stripped.

    :param s: The string from which the parentheses and enclosed text are to
        be removed.
    :type s: str
    :return: A new string with parentheses and their contents removed, extra
        spaces reduced, and leading/trailing spaces stripped.
    :rtype: str
    """
    return re.sub(r'\(.*\)', '', s).replace('  ', ' ').strip()


def parse_extracted_topics(raw_text: str) -> Tuple[TopicCollection, List[str]]:
    """
    Parses a raw text input to extract structured topics, propositions, entities,
    and their relationships. This function interprets specific formatted strings
    such as topics, propositions, entities, and their relationships to
    construct a collection of topics and supplemental unstructured data.

    :param raw_text: The raw text input to be parsed, containing topics,
        propositions, entities, and relationships in a predefined format.
    :type raw_text: str

    :return: A tuple containing the parsed topic collection and a list of
        unstructured or unparseable information from the input.
    :rtype: Tuple[TopicCollection, List[str]]
    """
    garbage = []
    current_state = None

    topics = TopicCollection(topics=[])

    current_topic = Topic(value=DEFAULT_TOPIC, facts=[], details=[])
    current_statement: Statement = None
    current_entities = {}

    for line in raw_text.split('\n'):

        if not line:
            continue

        line = line.strip()

        if line.startswith('topic:'):

            if current_statement and (
                current_statement.details or current_statement.facts
            ):
                current_topic.statements.append(current_statement)

            if current_entities:
                current_topic.entities = list(current_entities.values())

            if current_topic.entities or current_topic.statements:
                topics.topics.append(current_topic)

            current_state = None
            current_statement = None
            current_entities = {}

            topic_str = format_value(''.join(line.split(':')[1:]).strip())
            topic_str = strip_full_stop(topic_str)

            current_topic = Topic(value=topic_str, facts=[], details=[])

            continue

        if line.startswith('proposition:'):

            if current_statement and (
                current_statement.details or current_statement.facts
            ):
                current_topic.statements.append(current_statement)

            statement_str = format_value(''.join(line.split(':')[1:]).strip())
            current_statement = Statement(value=statement_str, facts=[], details=[])

            current_state = 'relationship-extraction'

            continue

        elif line.startswith('entities:'):
            current_state = 'entity-extraction'
            continue

        elif line.startswith('entity-') and line.endswith('relationships:'):
            current_state = 'relationship-extraction'
            continue

        elif current_state and current_state == 'entity-extraction':
            parts = line.split('|')
            if len(parts) == 2:
                entity_raw_value = parts[0]
                entity_clean_value = clean(entity_raw_value)
                entity = Entity(
                    value=entity_clean_value,
                    classification=format_classification(parts[1]),
                )
                if entity_clean_value not in current_entities:
                    current_entities[entity_clean_value] = entity
            else:
                garbage.append(f'UNPARSEABLE ENTITY: {line}')

        elif current_state and current_state == 'relationship-extraction':
            parts = line.split('|')
            fact = None
            if len(parts) == 3:
                s, p, o = parts
                if s and p and o:
                    s_entity = current_entities.get(clean(s), None)
                    o_entity = current_entities.get(clean(o), None)
                    if s_entity and o_entity:
                        fact = Fact(
                            subject=s_entity,
                            predicate=Relation(value=format_value(p)),
                            object=o_entity,
                        )
                        if current_statement:
                            current_statement.facts.append(fact)
                    elif s_entity:
                        fact = Fact(
                            subject=s_entity,
                            predicate=Relation(value=format_value(p)),
                            complement=format_value(o),
                        )
                        if current_statement:
                            current_statement.facts.append(fact)

            if not fact:
                if parts and current_statement:
                    details = ' '.join([format_value(part) for part in parts])
                    if details:
                        current_statement.details.append(details)
                garbage.append(f'STATEMENT DETAIL: {line}')

        else:
            garbage.append(f'UNPARSEABLE: {line}')

    if current_topic:
        if current_statement and (current_statement.details or current_statement.facts):
            current_topic.statements.append(current_statement)

        if current_entities:
            current_topic.entities = list(current_entities.values())

        if current_topic.entities or current_topic.statements:
            topics.topics.append(current_topic)

    if logger.isEnabledFor(logging.DEBUG):
        s = f"""====================================
raw_text: {raw_text}
------------------------------------
topics: {topics}
------------------------------------
garbage: {garbage}
"""
        logger.debug(s)

    return (topics, garbage)
