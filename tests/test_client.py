"""Tests for pac_cortex.client — VmClient and HarnessClient."""

from unittest.mock import MagicMock, patch

import pytest
from connectrpc.code import Code as _ConnectCode

from pac_cortex.client import HarnessClient, Task, Trial, TrialResult, VmClient


class _ConnectError(Exception):
    """Fake ConnectError for patching connectrpc.errors.ConnectError."""

    code = _ConnectCode.UNAVAILABLE
    message = "server down"


# ---------------------------------------------------------------------------
# VmClient
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_vm():
    """VmClient with PcmRuntimeClientSync and _settings patched."""
    with (
        patch("pac_cortex.client.PcmRuntimeClientSync") as mock_pcm_cls,
        patch("pac_cortex.client._settings") as mock_settings,
    ):
        mock_settings.vm_call_retries = 0
        mock_settings.vm_call_timeout_s = 5.0
        mock_pcm = MagicMock()
        mock_pcm_cls.return_value = mock_pcm
        yield VmClient("http://localhost:8080"), mock_pcm


def test_vm_read_forwards_path(patched_vm) -> None:
    """read() forwards path to the stub and returns MessageToDict result."""
    vm, mock_pcm = patched_vm
    fake_proto = MagicMock()
    mock_pcm.read.return_value = fake_proto

    with patch("pac_cortex.client.MessageToDict", return_value={"content": "hello"}) as mock_dict:
        result = vm.read("/etc/hosts")

    mock_dict.assert_called_once_with(fake_proto)
    assert result == {"content": "hello"}


def test_vm_call_none_result_returns_empty_dict(patched_vm) -> None:
    """_call() returns {} when stub returns None (proto empty message)."""
    vm, mock_pcm = patched_vm
    mock_pcm.tree.return_value = None

    result = vm.tree("/")

    assert result == {}


def test_vm_call_connect_error_propagates(patched_vm) -> None:
    """ConnectError from stub propagates after retries=0."""
    vm, mock_pcm = patched_vm
    mock_pcm.read.side_effect = _ConnectError()

    with (
        patch("pac_cortex.client.ConnectError", _ConnectError),
        pytest.raises(_ConnectError),
    ):
        vm.read("/foo")


def test_vm_call_timeout_raises_runtime_error(patched_vm) -> None:
    """TimeoutError from future.result() is converted to RuntimeError after retries=0."""
    vm, _ = patched_vm

    with patch("pac_cortex.client.ThreadPoolExecutor") as mock_exec_cls:
        mock_exec = MagicMock()
        mock_exec_cls.return_value.__enter__.return_value = mock_exec
        mock_future = MagicMock()
        mock_exec.submit.return_value = mock_future
        mock_future.result.side_effect = TimeoutError()

        with pytest.raises(RuntimeError, match="timed out"):
            vm.read("/slow")


def test_vm_answer_maps_outcome_and_forwards_refs(patched_vm) -> None:
    """answer() constructs AnswerRequest with correct fields."""
    vm, mock_pcm = patched_vm
    mock_pcm.answer.return_value = MagicMock()

    with patch("pac_cortex.client.MessageToDict", return_value={}):
        vm.answer("task done", "OUTCOME_OK", refs=["path/a", "path/b"])

    req = mock_pcm.answer.call_args[0][0]
    assert req.message == "task done"
    assert list(req.refs) == ["path/a", "path/b"]


def test_vm_answer_unknown_outcome_raises_key_error(patched_vm) -> None:
    """Unknown outcome name raises KeyError before any network call."""
    vm, mock_pcm = patched_vm

    with pytest.raises(KeyError):
        vm.answer("oops", "OUTCOME_BOGUS")

    mock_pcm.answer.assert_not_called()


def test_vm_call_retries_on_connect_error(patched_vm) -> None:
    """With retries=1, ConnectError on first attempt is retried once."""
    vm, mock_pcm = patched_vm
    # Override retries to 1
    with patch("pac_cortex.client._settings") as s:
        s.vm_call_retries = 1
        s.vm_call_timeout_s = 5.0
        fake_proto = MagicMock()
        mock_pcm.list.side_effect = [_ConnectError(), fake_proto]

        with (
            patch("pac_cortex.client.ConnectError", _ConnectError),
            patch("pac_cortex.client.MessageToDict", return_value={"entries": []}),
            patch("pac_cortex.client.time.sleep"),
        ):
            result = vm.list("/")

    assert result == {"entries": []}
    assert mock_pcm.list.call_count == 2


# ---------------------------------------------------------------------------
# HarnessClient
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_harness():
    """HarnessClient with HarnessServiceClientSync patched."""
    with patch("pac_cortex.client.HarnessServiceClientSync") as mock_cls:
        mock_stub = MagicMock()
        mock_cls.return_value = mock_stub
        yield HarnessClient("http://harness"), mock_stub


def test_harness_list_tasks_returns_task_list(patched_harness) -> None:
    client, mock_stub = patched_harness
    t1, t2 = MagicMock(task_id="task-001"), MagicMock(task_id="task-002")
    mock_stub.get_benchmark.return_value = MagicMock(tasks=[t1, t2])

    tasks = client.list_tasks("bench-1")

    assert len(tasks) == 2
    assert all(isinstance(t, Task) for t in tasks)
    assert tasks[0].task_id == "task-001"
    assert tasks[1].task_id == "task-002"


def test_harness_start_trial_returns_trial(patched_harness) -> None:
    client, mock_stub = patched_harness
    mock_stub.start_playground.return_value = MagicMock(
        trial_id="trial-42",
        harness_url="http://vm-host",
        instruction="solve it",
    )

    trial = client.start_trial("bench-1", "task-001")

    assert isinstance(trial, Trial)
    assert trial.trial_id == "trial-42"
    assert trial.harness_url == "http://vm-host"
    assert trial.instruction == "solve it"


def test_harness_end_trial_returns_trial_result(patched_harness) -> None:
    client, mock_stub = patched_harness
    mock_stub.end_trial.return_value = MagicMock(score=0.85, score_detail=["ok", "partial"])

    result = client.end_trial("trial-42")

    assert isinstance(result, TrialResult)
    assert result.trial_id == "trial-42"
    assert result.score == 0.85
    assert result.score_detail == ["ok", "partial"]
