"""The single retrieval and agent backend: agent-core."""

import config
from evalkit.api_client import RagServiceClient


def make_client() -> RagServiceClient:
    return RagServiceClient(
        base_url=config.AGENT_PLATFORM_URL,
        api_key=config.AGENT_PLATFORM_TOKEN,
        timeout=config.RAGAS_TIMEOUT,
    )
