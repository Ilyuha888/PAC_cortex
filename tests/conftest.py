from unittest.mock import DEFAULT, MagicMock

import pytest

from pac_cortex.agent import AssembledPrompt
from pac_cortex.client import VmClient
from pac_cortex.llm import LLMClient

_DEFAULT_ASSEMBLED = AssembledPrompt(
    include_entity_inbox=False, vocabulary={}
)


@pytest.fixture
def mock_vm_client() -> MagicMock:
    mock = MagicMock(spec=VmClient)
    mock.tree.return_value = {"entries": []}
    mock.find.return_value = {"entries": []}
    mock.search.return_value = {"entries": []}
    mock.list.return_value = {"entries": []}
    mock.read.return_value = {"content": ""}
    mock.write.return_value = {}
    mock.delete.return_value = {}
    mock.mkdir.return_value = {}
    mock.move.return_value = {}
    mock.answer.return_value = {}
    return mock


@pytest.fixture
def mock_llm_client() -> MagicMock:
    mock = MagicMock(spec=LLMClient)
    mock.total_prompt_tokens = 0
    mock.total_completion_tokens = 0

    # Pre-flight calls parse_step(messages, AssembledPrompt, ...) once per task.
    # Return a neutral default so tests only need to configure NextStep responses.
    # Tests that override side_effect with a list must prepend an AssembledPrompt
    # as the first element to account for the pre-flight call.
    def _preflight_dispatch(
        messages, response_format, max_completion_tokens=16384, extra_body=None
    ):
        if response_format is AssembledPrompt:
            return _DEFAULT_ASSEMBLED
        return DEFAULT  # tells MagicMock to use configured return_value

    mock.parse_step.side_effect = _preflight_dispatch
    return mock
