#!/bin/bash
set -e

# ---- Required ----
PROFILE="${1:-default}"
TENANT_ID="${2:-default-tenant}"
PROMPT_ID="${3:-default-prompt}"
REGION="us-east-1"
TABLE_NAME="PromptTemplates"

# ---- Optional ----
SYSTEM_PROMPT_FILE=""
USER_PROMPT_FILE=""

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      SYSTEM_PROMPT_FILE="$2"
      shift 2
      ;;
    --user)
      USER_PROMPT_FILE="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$SYSTEM_PROMPT_FILE" && -z "$USER_PROMPT_FILE" ]]; then
  echo "Please provide at least --system <file> or --user <file>"
  exit 1
fi

get_format() {
  if [[ "$1" == *.json ]]; then
    echo "json"
  elif [[ "$1" == *.txt ]]; then
    echo "text"
  else
    echo "unknown"
  fi
}

FORMAT="text"
UPDATE_EXPRESSION="SET"
EXPR_ATTRS=()
EXPR_VALS=()

if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  [[ ! -f "$SYSTEM_PROMPT_FILE" ]] && echo "File not found: $SYSTEM_PROMPT_FILE" && exit 1
  FORMAT=$(get_format "$SYSTEM_PROMPT_FILE")
  CONTENT=$( [[ "$FORMAT" == "json" ]] && jq -c . < "$SYSTEM_PROMPT_FILE" || jq -Rs . < "$SYSTEM_PROMPT_FILE" )
  UPDATE_EXPRESSION="$UPDATE_EXPRESSION SystemPrompt = :sp,"
  EXPR_VALS+=("sp=$CONTENT")
fi

if [[ -n "$USER_PROMPT_FILE" ]]; then
  [[ ! -f "$USER_PROMPT_FILE" ]] && echo "File not found: $USER_PROMPT_FILE" && exit 1
  FORMAT=$(get_format "$USER_PROMPT_FILE")
  CONTENT=$( [[ "$FORMAT" == "json" ]] && jq -c . < "$USER_PROMPT_FILE" || jq -Rs . < "$USER_PROMPT_FILE" )
  UPDATE_EXPRESSION="$UPDATE_EXPRESSION UserPrompt = :up,"
  EXPR_VALS+=("up=$CONTENT")
fi

UPDATE_EXPRESSION="$UPDATE_EXPRESSION Format = :fmt"
EXPR_VALS+=("fmt=\"$FORMAT\"")

# ---- Construct CLI arguments ----
declare -A attr_vals
for pair in "${EXPR_VALS[@]}"; do
  IFS='=' read -r key val <<< "$pair"
  attr_vals[$key]=$val
done

EXPR_JSON=$(jq -n --arg sp "${attr_vals[sp]}" --arg up "${attr_vals[up]}" --arg fmt "${attr_vals[fmt]}" '
  {
    ":sp": ($sp | select(. != null) | fromjson)?,
    ":up": ($up | select(. != null) | fromjson)?,
    ":fmt": { "S": $fmt }
  } | with_entries(select(.value != null))
')

# ---- Update Item ----
aws dynamodb update-item \
  --region "$REGION" \
  --profile "$PROFILE" \
  --table-name "$TABLE_NAME" \
  --key "{\"TenantID\": {\"S\": \"$TENANT_ID\"}, \"PromptID\": {\"S\": \"$PROMPT_ID\"}}" \
  --update-expression "$UPDATE_EXPRESSION" \
  --expression-attribute-values "$EXPR_JSON"

echo "Updated prompt '$PROMPT_ID' (Tenant: $TENANT_ID) with format '$FORMAT'"
