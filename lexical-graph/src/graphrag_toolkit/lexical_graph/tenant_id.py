# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union
from llama_index.core.bridge.pydantic import BaseModel


class TenantId(BaseModel):
    """Represents a tenant-specific identifier with utility methods to manage and format
    tenant-related data.

    This class is used for handling tenant-specific data, allowing for tenant-specific
    identification, formatted output, and unique namespace handling. It ensures that tenant
    values adhere to specific validation criteria and provides utility methods for formatting
    labels, indexes, IDs, and hashable strings. It also supports determining whether the
    tenant is a default tenant.

    :ivar value: The tenant identifier string. Must be between 1-10 characters,
                 alphanumeric, entirely lowercase, or None for the default tenant.
    :type value: Optional[str]
    """

    value: Optional[str] = None

    def __init__(self, value: str = None):
        """
        Initializes a new instance of the class with the given value, validating the
        constraints on the value format. The value must be a string containing between 1
        and 10 characters, consisting of lowercase letters or numbers. Uppercase letters
        or invalid formats will cause a ValueError to be raised.

        :param value: The tenant ID value. Defaults to None if not provided.
        :type value: str, optional

        :raises ValueError: If the value provided does not adhere to the specified constraints.
        """
        if value is not None:
            if (
                len(value) > 10
                or len(value) < 1
                or not value.isalnum()
                or any(letter.isupper() for letter in value)
            ):
                raise ValueError(
                    f"Invalid TenantId: '{value}'. TenantId must be between 1-10 lowercase letters and numbers."
                )
        super().__init__(value=value)

    def is_default_tenant(self):
        """
        Determines whether the current tenant is the default tenant.

        This method checks if the `value` attribute is set to None, which indicates
        that the current instance represents the default tenant.

        :return: Boolean indicating whether the tenant is the default tenant.
        :rtype: bool
        """
        return self.value is None

    def format_label(self, label: str):
        """
        Formats the provided label based on the tenant configuration. If the
        default tenant is being used, the label will remain unchanged
        except for being wrapped with backticks. Otherwise, the label
        will be suffixed with the tenant's value, followed by two
        underscores, and wrapped with backticks.

        :param label: The label string to be formatted.
        :type label: str
        :return: The formatted label string based on tenant configuration.
        :rtype: str
        """
        if self.is_default_tenant():
            return f'`{label}`'
        return f'`{label}{self.value}__`'

    def format_index_name(self, index_name: str):
        """
        Formats the provided index name based on the tenant context.

        If the current tenant is the default tenant, the method returns the
        index name as is. Otherwise, it appends the tenant value to the index
        name.

        :param index_name: The name of the index to be formatted.
        :type index_name: str
        :return: The formatted index name, either unchanged or appended with
            the tenant value.
        :rtype: str
        """
        if self.is_default_tenant():
            return index_name
        return f'{index_name}_{self.value}'

    def format_hashable(self, hashable: str):
        """
        Formats a given hashable string by appending a namespace prefix when the tenant
        is not the default. This adds a unique identifier to the string to distinguish
        it in multi-tenant environments.

        :param hashable: The hashable string to format.
        :type hashable: str
        :return: A formatted hashable string. If the tenant is default, the original
            string is returned. Otherwise, it returns the string prefixed with the
            tenant's value followed by '::'.
        :rtype: str
        """
        if self.is_default_tenant():
            return hashable
        else:
            return f'{self.value}::{hashable}'

    def format_id(self, prefix: str, id_value: str):
        """
        Formats an identifier string with a specified prefix and id_value, adhering to a specific
        format depending on the tenant configuration. Provides support for default tenant cases
        and non-default tenant cases by dynamically modifying the format of the resulting string.

        :param prefix: The prefix to prepend to the formatted identifier, indicating the
                       category or type of the identifier.
        :type prefix: str
        :param id_value: The main identifier value to be included in the formatted string,
                         representing the unique ID.
        :type id_value: str
        :return: A formatted string integrating the prefix, tenant status, and id_value that
                 complies with the required structure.
        :rtype: str
        """
        if self.is_default_tenant():
            return f'{prefix}::{id_value}'
        else:
            return f'{prefix}:{self.value}:{id_value}'

    def rewrite_id(self, id_value: str):
        """
        Rewrites the given ID by incorporating the value of the current tenant if it is not the default tenant.

        The method checks whether the current tenant is the default tenant. If so, the original
        ID value is returned unchanged. Otherwise, it modifies the ID by inserting the tenant's
        value into the appropriate position within the ID.

        :param id_value: The original ID value in string format.
        :type id_value: str
        :return: The modified ID if the tenant is not default, or the original ID if it is.
        :rtype: str
        """
        if self.is_default_tenant():
            return id_value
        else:
            id_parts = id_value.split(':')
            return f'{id_parts[0]}:{self.value}:{":".join(id_parts[2:])}'


DEFAULT_TENANT_ID = TenantId()

TenantIdType = Union[str, TenantId]


def to_tenant_id(tenant_id: Optional[TenantIdType]):
    """
    Convert the provided tenant identifier to a TenantId instance. This function ensures
    that a valid TenantId object is always returned, regardless of the input's type, provided
    it is convertible to a string. If the input is `None`, a default TenantId is returned.

    :param tenant_id: The tenant identifier that can be of type TenantIdType or None.
                      If None, the default tenant ID is returned.
    :type tenant_id: Optional[TenantIdType]

    :return: A `TenantId` object based on the provided `tenant_id`. If `tenant_id` is None,
             it returns the default tenant ID. If `tenant_id` is an instance of `TenantId`,
             it returns the same object. Otherwise, it converts `tenant_id` to string and
             creates a new TenantId object.
    :rtype: TenantId
    """
    if tenant_id is None:
        return DEFAULT_TENANT_ID
    if isinstance(tenant_id, TenantId):
        return tenant_id
    else:
        return TenantId(str(tenant_id))
