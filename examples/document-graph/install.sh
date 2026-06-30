#!/bin/bash
# Lifecycle config — must finish in <5 min. Do absolute minimum.
set -e

# Configure these for your environment:
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-<insert-account-number-here>}"
ARTIFACTS_BUCKET="${ARTIFACTS_BUCKET:-graphrag-artifacts-${AWS_ACCOUNT_ID}}"

aws s3 sync "s3://${ARTIFACTS_BUCKET}/document-graph-notebooks/" ~/SageMaker/document-graph/ --quiet --exclude "*.ipynb_checkpoints/*"
echo "done" > ~/SageMaker/document-graph/.ready
