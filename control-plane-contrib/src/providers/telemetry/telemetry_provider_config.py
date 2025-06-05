# control-plane-contrib/enrollment/enrollment_provider_config.py

from pydantic import BaseModel
from typing import Optional

class EnrollmentFormatMixin(BaseModel):
    """
    Mixin to support different enrollment format types (e.g., JSON or text).
    """
    format: Optional[str] = "json"  # Options: "json", "text"

class FileEnrollmentProviderConfig(EnrollmentFormatMixin, BaseModel):
    """
    Configuration for loading enrollment data from local filesystem.
    """
    base_path: str
    enrollment_file: str = "enrollment.json"

class S3EnrollmentProviderConfig(EnrollmentFormatMixin, BaseModel):
    """
    Configuration for loading enrollment data from Amazon S3.
    """
    bucket: str
    key: str
    region: Optional[str] = None
    profile_name: Optional[str] = None

class DynamoDBEnrollmentProviderConfig(EnrollmentFormatMixin, BaseModel):
    """
    Configuration for loading enrollment data from DynamoDB.
    """
    table_name: str
    tenant_id: str
    enrollment_key: str
    region: Optional[str] = None
    profile_name: Optional[str] = None
