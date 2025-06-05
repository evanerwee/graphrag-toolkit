#!/bin/bash

set -e

# Configuration
TABLE_NAME="PromptTemplates"
REGION="us-east-1"
PROFILE="${1:-default}"  # Use first argument or default profile

echo "Using AWS CLI profile: $PROFILE"
echo "Creating DynamoDB table: $TABLE_NAME in region: $REGION"

# Create the table
aws dynamodb create-table \
    --region "$REGION" \
    --profile "$PROFILE" \
    --table-name "$TABLE_NAME" \
    --attribute-definitions \
        AttributeName=TenantID,AttributeType=S \
        AttributeName=PromptID,AttributeType=S \
    --key-schema \
        AttributeName=TenantID,KeyType=HASH \
        AttributeName=PromptID,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --table-class STANDARD \
    --tags Key=Environment,Value=Dev

echo "DynamoDB table '$TABLE_NAME' created successfully using profile '$PROFILE'."
