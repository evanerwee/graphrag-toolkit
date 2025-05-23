# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime


def get_properties_str(properties, default):
    """
    Generates a string representation of the specified properties dictionary.

    This function takes a dictionary of properties, converts it into a sorted list
    of key-value pairs formatted as "key:value", and joins these pairs into a
    single string separated by semicolons. If the provided properties dictionary
    is empty or None, the function returns the specified default string.

    :param properties:
        Dictionary where keys and values represent property names and
        their corresponding values. Can be None.
    :param default:
        Default string to return if the properties dictionary is empty
        or None.
    :return:
        A semicolon-separated string representing the properties if they
        exist, or the default string if the properties dictionary is
        empty or None.
    :rtype:
        str
    """
    if properties:
        return ';'.join(sorted([f'{k}:{v}' for k, v in properties.items()]))
    else:
        return default


def last_accessed_date(*args):
    """
    Calculates and returns the current date formatted as a string in 'YYYY-MM-DD'
    format and associates it with the key 'last_accessed_date' in the returned
    dictionary.

    :param args: Optional arguments that are not utilized in the function.
    :type args: Any
    :return: A dictionary containing the current date under the key 'last_accessed_date'.
    :rtype: dict
    """
    return {'last_accessed_date': datetime.datetime.now().strftime("%Y-%m-%d")}
