from unittest.mock import MagicMock

import pytest

from pac_cortex.client import VmClient
from pac_cortex.llm import LLMClient


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
    return mock
