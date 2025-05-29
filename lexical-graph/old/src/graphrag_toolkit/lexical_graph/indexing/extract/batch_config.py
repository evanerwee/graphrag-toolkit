# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BatchConfig:
    """
    Configuration for batch processing.

    This class represents configurations required for batch processing, including
    details about AWS resources such as role ARN, region, bucket configuration,
    and processing limits. These configurations enable the setup and management
    of batch processing operations.

    :ivar role_arn: The ARN of the IAM role used for executing batch jobs.
    :type role_arn: str
    :ivar region: The AWS region where the batch process is executed.
    :type region: str
    :ivar bucket_name: The name of the S3 bucket to be used in the batch process.
    :type bucket_name: str
    :ivar key_prefix: (Optional) The prefix for keys in the S3 bucket used for
        batching.
    :type key_prefix: Optional[str]
    :ivar s3_encryption_key_id: (Optional) The ID of the encryption key used for
        encrypting data in S3.
    :type s3_encryption_key_id: Optional[str]
    :ivar subnet_ids: The list of subnet IDs for the network configuration.
    :type subnet_ids: List[str]
    :ivar security_group_ids: The list of security group IDs for the network
        configuration.
    :type security_group_ids: List[str]
    :ivar max_batch_size: The maximum number of entries allowed in a single batch.
        Default is 25,000.
    :type max_batch_size: int
    :ivar max_num_concurrent_batches: The maximum number of batches allowed to
        run concurrently. Default is 3.
    :type max_num_concurrent_batches: int
    """

    role_arn: str
    region: str
    bucket_name: str
    key_prefix: Optional[str] = None
    s3_encryption_key_id: Optional[str] = None
    subnet_ids: List[str] = field(default_factory=list)
    security_group_ids: List[str] = field(default_factory=list)
    max_batch_size: int = 25000
    max_num_concurrent_batches: int = 3
