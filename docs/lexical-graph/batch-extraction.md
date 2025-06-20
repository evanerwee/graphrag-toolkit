
[[Home](./)]

## Batch Extraction

### Topics

  - [Overview](#overview)
  - [Using batch inference with the LexicalGraphIndex](#using-batch-inference-with-the-lexicalgraphindex)
  - [Setup](#setup)
  - [Batch extraction job requirements](#batch-extraction-job-requirements)
  - [Configuring batch extraction](#configuring-batch-extraction)

### Overview

You can use [Amazon Bedrock batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) in the extract stage of the indexing process to improve the performance of extraction on large datasets.

### Using batch inference with the LexicalGraphIndex

To use batch inference in the extract stage of the indexing process, create a `BatchConfig` object and supply it to the `LexicalGraphIndex` as part of the [`IndexingConfig`](./indexing.md#configuring-the-extract-and-build-stages): 

```python
import os

from graphrag_toolkit.lexical_graph import LexicalGraphIndex
from graphrag_toolkit.lexical_graph import GraphRAGConfig, IndexingConfig
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.indexing.extract import BatchConfig

from llama_index.core import SimpleDirectoryReader
    
def batch_extract_and_load():
    
    GraphRAGConfig.extraction_batch_size = 100

    batch_config = BatchConfig(
        region='us-west-2',
        bucket_name='my-bucket',
        key_prefix='batch-extract',
        role_arn='arn:aws:iam::111111111111:role/my-batch-inference-role'
    )

    indexing_config = IndexingConfig(batch_config=batch_config)

    graph_store = GraphStoreFactory.for_graph_store(os.environ['GRAPH_STORE'])
    vector_store = VectorStoreFactory.for_vector_store(os.environ['VECTOR_STORE'])

    graph_index = LexicalGraphIndex(
        graph_store, 
        vector_store,
        indexing_config=indexing_config
    )

    reader = SimpleDirectoryReader(input_dir='path/to/directory')
    docs = reader.load_data()

    graph_index.extract_and_build(docs, show_progress=True)
    
batch_extract_and_load()
```

When using batch extraction, update the `GraphRAGConfig.extraction_batch_size` configuration parameter so that a large number of source documents are passed to a batch inference job in a single batch. In the example above, `GraphRAGConfig.extraction_batch_size` has been set to `100`, meaning that 100 source documents will be chunked simultaneously, and these chunks then sent to the batch inference job. If there are 10-20 chunks per document, the batch inference job here will process several thousand records in a single batch.

### Setup

Before running batch extraction for the first time, you must fulfill the following prerequisites:

  - Create an Amazon S3 bucket in the AWS Region where you will be running batch extraction
  - [Create a custom service role for batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html) with access to the S3 bucket (and permission to invoke an inference profile, if necessary)
  - Update the IAM identity under which the indexing process runs to allow it to to [submit and manage batch inference jobs](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-prereq.html#batch-inference-permissions) and pass the custom serice role to Bedrock

In the examples below, replace `<account-id>` with your AWS account ID, `<region>` with the name of the AWS Region where you will be running batch extraction, `<model-id>` with the ID of the foundation model in Amazon Bedrock that you want to use for batch extraction, and `<custom-service-role-arn>` with the ARN of your new custom service role.

#### Custom service role

[Create a custom service role for batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html) with the following trust relationship:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "<account-id>"
                },
                "ArnEquals": {
                    "aws:SourceArn": "arn:aws:bedrock:<region>:<account-id>:model-invocation-job/*"
                }
            }
        }
    ]
}
```

Create and attach a policy to your custom service role that [allows access to the Amazon S3 bucket where batch inference input and output files will be stored](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html#batch-iam-sr-identity):

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::<bucket>",
                "arn:aws:s3:::<bucket>/*"
            ],
            "Condition": {
                "StringEquals": {
                    "aws:ResourceAccount": [
                        "<account-id>"
                    ]
                }
             }
        }
    ]
}
```

To run batch inference with an inference profile, the service role [must have permissions to invoke the inference profile in an AWS Region](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html#batch-iam-sr-ip), in addition to the model in each Region in the inference profile.

#### Update IAM identity

You will also need to update the IAM identity under which the indexing process runs (not the custom service role) to allow it to to [submit and manage batch inference jobs](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-prereq.html#batch-inference-permissions): 

```
{
    "Version": "2012-10-17",
    "Statement": [
        ...
        
        {
            "Effect": "Allow",
            "Action": [  
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:ListModelInvocationJobs",
                "bedrock:StopModelInvocationJob"
            ],
            "Resource": [
                "arn:aws:bedrock:<region>::foundation-model/<model-id>",
                "arn:aws:bedrock:<region>:<account-id>:model-invocation-job/*"
            ]
        }
    ]
}
```

Add the `iam:PassRole` permission so that the IAM identity under which the indexing process runs can pass the custom service role to Bedrock:

```
{
    "Effect": "Allow",
    "Action": [
        "iam:PassRole"
    ],
    "Resource": "<custom-service-role-arn>"
}
```

### Batch extraction job requirements

Each batch extraction job must follow Amazon Bedrock's [batch inference quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-data.html). The lexical-graph's batch extraction uses one input file per job.

#### Key requirements

  - Each batch job needs 100-50,000 records
  - Jobs with fewer than 100 records are processed individually, not in batch
  - The feature doesn't check input file sizes — jobs will fail if they exceed Bedrock quotas

#### Worker configuration
Batch extraction can use multiple workers that trigger concurrent batch jobs:

  - If (workers × concurrent batches) exceeds Bedrock quotas, jobs will wait until capacity is available

#### Important configuration settings

  - `GraphRAGConfig.extraction_batch_size`: Sets how many source documents go to the extraction pipeline. Ensure (source documents × average chunks per document) is enough to fill your planned simultaneous batch jobs.
  - `GraphRAGConfig.extraction_num_workers`: Sets how many CPUs run batch jobs simultaneously.
  - `BatchConfig.max_num_concurrent_batches`: Sets how many concurrent batch jobs each worker runs.
  - `BatchConfig.max_batch_size`: Sets the maximum number of chunks per batch job.

### Configuring batch extraction

The `BatchConfig` object has the following parameters:

| Parameter  | Description | Mandatory | Default Value |
| ------------- | ------------- | ------------- | ------------- |
| `bucket_name` | Name of an Amazon S3 bucket where batch input and output files will be stored | Y | |
| `region` | The name of the AWS Region in which the bucket is located and the Amazon Bedrock batch inference job will run (e.g. us-east-1) | Y | |
| `role_arn` | The Amazon Resource Name (ARN) of the service role with permissions to carry out and manage batch inference (you can use the console to create a default service role or follow the steps at [Create a service role for batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html)) | Y | |
| `key_prefix` | S3 key prefix for input and output files | N | |
| `max_batch_size` | Maximun number of records (chunks) to be included in each batch sent to each batch inference job | N | `25000` |
| `max_num_concurrent_batches` | Maximum number of batch inference jobs to run concurrently per _worker_ (see [`GraphRAGConfig.extraction_num_workers`](./configuration.md#graphragconfig)) | N | `3` |
| `s3_encryption_key_id` | The unique identifier of the key that encrypts the S3 location of the output data. | N | |
| `subnet_ids` | An array of IDs for each subnet in the Virtual Private Cloud (VPC) used to protect batch inference jobs (for more information, see [Protect batch inference jobs using a VPC](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-vpc))| N | |
| `security_group_ids` | An array of IDs for each security group in the Virtual Private Cloud (VPC) used to protect batch inference jobs (for more information, see [Protect batch inference jobs using a VPC](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-vpc))| N | |
| `delete_on_success` | Delete the input and output JSON files from the local filesystem on successful completion of a batch job. Input and output files in S3 are not deleted. | N | `True` |

#### Controlling access to batch extraction data

The `BatchConfig` allows you to specify a custom KMS key to encrypt the data in S3, and supply VPC subnet and security group ids to [protect batch inference jobs using a VPC](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-vpc).