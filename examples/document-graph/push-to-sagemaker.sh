#!/bin/bash
# push-to-sagemaker.sh — Rebuild wheel + upload to S3 (SageMaker pulls from there)
# Run locally after making changes to document-graph.
# Then in SageMaker terminal: bash ~/SageMaker/document-graph/install.sh
set -e

# Configure these for your environment:
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-<insert-account-number-here>}"
ARTIFACTS_BUCKET="${ARTIFACTS_BUCKET:-graphrag-artifacts-${AWS_ACCOUNT_ID}}"
AWS_REGION="${AWS_REGION:-us-east-1}"

cd "$(dirname "$0")/../.."
export AWS_PROFILE="${AWS_PROFILE:-default}"

echo "Building wheel..."
rm -f dist/*.whl
python3 -m build --wheel --outdir dist/ 2>&1 | tail -1

echo "Uploading wheel..."
aws s3 cp dist/graphrag_toolkit_document_graph-*.whl "s3://${ARTIFACTS_BUCKET}/document-graph-notebooks/wheels/" --region "${AWS_REGION}"

echo "Cleaning S3 notebooks (removing stale files)..."
aws s3 rm "s3://${ARTIFACTS_BUCKET}/document-graph-notebooks/" --recursive --exclude "wheels/*" --region "${AWS_REGION}"

echo "Uploading notebooks..."
aws s3 sync examples/document-graph/notebooks/ "s3://${ARTIFACTS_BUCKET}/document-graph-notebooks/" \
  --exclude "__pycache__/*" --exclude "*.pyc" --exclude "wheels/*" --region "${AWS_REGION}"

echo ""
echo "✅ Pushed. In SageMaker run:"
echo "   aws s3 sync s3://${ARTIFACTS_BUCKET}/document-graph-notebooks/ ~/SageMaker/document-graph/"
echo "   pip install --force-reinstall ~/SageMaker/document-graph/wheels/graphrag_toolkit_document_graph-*.whl"
