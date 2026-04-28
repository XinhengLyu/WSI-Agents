"""Tests for TaskAllocationAgent (smoke-only — handler hits LLM)."""
from unittest.mock import MagicMock

from agent import TaskAllocationAgent


def test_task_allocation_agent_constructs(agent_ctx):
    """Agent should construct without exception with a mocked client."""
    with agent_ctx():
        agent = TaskAllocationAgent(model_client=MagicMock())
    assert agent._model_client is not None
    # System message should mention the four agent types
    content = agent._system_message.content
    for keyword in ("Morphology", "Diagnosis", "Treatment", "Report"):
        assert keyword in content
