# control-plane-contrib/enrollment/enrollment_provider_factory.py

import os
from control_plane_contrib.enrollment.enrollment_provider_base import EnrollmentProvider
from control_plane_contrib.enrollment.enrollment_provider_config import (
    FileEnrollmentProviderConfig,
    S3EnrollmentProviderConfig,
    DynamoDBEnrollmentProviderConfig,
)
from control_plane_contrib.enrollment.providers.file_enrollment_provider import FileEnrollmentProvider
from control_plane_contrib.enrollment.providers.s3_enrollment_provider import S3EnrollmentProvider
from control_plane_contrib.enrollment.providers.dynamodb_enrollment_provider import DynamoDBEnrollmentProvider

from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class EnrollmentProviderFactory:
    """
    Factory for dynamically instantiating EnrollmentProvider implementations
    based on environment configuration.
    """

    @staticmethod
    def get_provider() -> EnrollmentProvider:
        """
        Determines the appropriate provider type from the ENROLLMENT_PROVIDER environment variable
        and returns an instantiated EnrollmentProvider.

        Returns:
            EnrollmentProvider: The configured provider implementation.
        """
        provider_type = os.getenv("ENROLLMENT_PROVIDER", "file").lower()
        logger.info(f"[Enrollment Debug] Selected provider: {provider_type}")

        if provider_type == "s3":
            config = S3EnrollmentProviderConfig()  # In practice, pass args or load from file/env
            return S3EnrollmentProvider(config)

        elif provider_type == "dynamodb":
            config = DynamoDBEnrollmentProviderConfig()
            return DynamoDBEnrollmentProvider(config)

        else:  # default to 'file'
            config = FileEnrollmentProviderConfig(base_path="./enrollments")
            return FileEnrollmentProvider(config)
