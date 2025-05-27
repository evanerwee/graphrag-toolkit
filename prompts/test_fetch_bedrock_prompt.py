import boto3
import argparse
import json

def fetch_prompt_text(prompt_identifier: str, region: str, profile: str = None):
    # Initialize boto3 session using specified profile
    session = boto3.Session(profile_name=profile, region_name=region)

    bedrock_agent = session.client("bedrock-agent")

    # If short name, convert to full ARN
    if not prompt_identifier.startswith("arn:"):
        sts = session.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        prompt_identifier = f"arn:aws:bedrock:{region}:{account_id}:prompt/{prompt_identifier}"

    try:
        response = bedrock_agent.get_prompt(promptIdentifier=prompt_identifier)

        print("\n✅ Raw response:")
        print(json.dumps(response, indent=2, default=str))  # Fix datetime serialization

        # Attempt to extract full prompt text
        text = response["variants"][0]["templateConfiguration"]["text"]["text"]
        print("\n✅ Extracted prompt text:")
        print(text)

    except Exception as e:
        print(f"\n❌ Failed to fetch or parse prompt: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch prompt text from Bedrock Agent.")
    parser.add_argument("--prompt", required=True, help="Prompt name or full ARN")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", required=True, help="AWS CLI profile name")

    args = parser.parse_args()
    fetch_prompt_text(args.prompt, args.region, args.profile)
