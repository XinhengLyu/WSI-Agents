import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import Dict, Optional

# API configuration — set via environment variables, or provide defaults here.
# For open-source release, set OPENAI_API_KEY / OPENAI_BASE_URL in your environment.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

# Default model capabilities
DEFAULT_MODEL_CAPABILITIES = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
}

def create_model_client(
        model_name: str,
        model_capabilities: Optional[Dict] = None
) -> OpenAIChatCompletionClient:
    """
    Create an OpenAI chat completion client with default configuration.

    Args:
        model_name: The name of the model to use
        model_capabilities: Optional dict of model capabilities, will use defaults if not provided

    Returns:
        OpenAIChatCompletionClient instance
    """
    capabilities = model_capabilities if model_capabilities is not None else DEFAULT_MODEL_CAPABILITIES

    return OpenAIChatCompletionClient(
        model=model_name,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_capabilities=capabilities,
    )
