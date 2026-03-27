"""BitGN API clients: harness lifecycle and per-task VM runtime."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bitgn.harness_connect import HarnessServiceClientSync  # noqa: I001
from bitgn.harness_pb2 import (
    EndTrialRequest,
    GetBenchmarkRequest,
    StartPlaygroundRequest,
    StatusRequest,
)
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import (
    AnswerRequest,
    ContextRequest,
    DeleteRequest,
    FindRequest,
    ListRequest,
    MkDirRequest,
    MoveRequest,
    Outcome,
    ReadRequest,
    SearchRequest,
    TreeRequest,
    WriteRequest,
)
from connectrpc.code import Code as _ConnectCode
from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

from pac_cortex.config import settings as _settings

logger = logging.getLogger(__name__)

# ConnectRPC error codes that are transient and worth retrying.
# Deterministic codes (NOT_FOUND, INVALID_ARGUMENT, PERMISSION_DENIED, …) are not retried.
_TRANSIENT_CODES = {
    _ConnectCode.UNAVAILABLE,
    _ConnectCode.DEADLINE_EXCEEDED,
    _ConnectCode.UNKNOWN,
    _ConnectCode.INTERNAL,
}

# ---------------------------------------------------------------------------
# Typed boundary models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    task_id: str


class Trial(BaseModel):
    trial_id: str
    harness_url: str
    instruction: str


class TrialResult(BaseModel):
    trial_id: str
    score: float
    score_detail: list[str]


# ---------------------------------------------------------------------------
# Internal lookup table — kept in client.py so Outcome never escapes
# ---------------------------------------------------------------------------

_OUTCOME_BY_NAME: dict[str, Any] = {
    "OUTCOME_OK": Outcome.OUTCOME_OK,
    "OUTCOME_DENIED_SECURITY": Outcome.OUTCOME_DENIED_SECURITY,
    "OUTCOME_NONE_CLARIFICATION": Outcome.OUTCOME_NONE_CLARIFICATION,
    "OUTCOME_NONE_UNSUPPORTED": Outcome.OUTCOME_NONE_UNSUPPORTED,
    "OUTCOME_ERR_INTERNAL": Outcome.OUTCOME_ERR_INTERNAL,
}


# ---------------------------------------------------------------------------
# Harness client — benchmark lifecycle
# ---------------------------------------------------------------------------

class HarnessClient:
    """Control-plane client for benchmark orchestration."""

    def __init__(self, host: str) -> None:
        self._client = HarnessServiceClientSync(host)

    def status(self) -> str:
        return str(self._client.status(StatusRequest()))

    def list_tasks(self, benchmark_id: str) -> list[Task]:
        res = self._client.get_benchmark(GetBenchmarkRequest(benchmark_id=benchmark_id))
        return [Task(task_id=t.task_id) for t in res.tasks]

    def start_trial(self, benchmark_id: str, task_id: str) -> Trial:
        trial = self._client.start_playground(
            StartPlaygroundRequest(benchmark_id=benchmark_id, task_id=task_id)
        )
        return Trial(
            trial_id=trial.trial_id,
            harness_url=trial.harness_url,
            instruction=trial.instruction,
        )

    def end_trial(self, trial_id: str) -> TrialResult:
        result = self._client.end_trial(EndTrialRequest(trial_id=trial_id))
        return TrialResult(
            trial_id=trial_id,
            score=result.score,
            score_detail=list(result.score_detail),
        )


# ---------------------------------------------------------------------------
# VM client — per-task filesystem runtime
# ---------------------------------------------------------------------------

class VmClient:
    """Per-task PCM filesystem runtime client."""

    def __init__(self, harness_url: str) -> None:
        self._vm = PcmRuntimeClientSync(harness_url)

    def _call(self, fn: Callable[[], Any]) -> dict[str, Any]:
        retries = _settings.vm_call_retries
        for attempt in range(retries + 1):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fn)
                try:
                    result = future.result(timeout=_settings.vm_call_timeout_s)
                    return MessageToDict(result) if result else {}
                except TimeoutError:
                    logger.warning("VmClient call timed out (attempt %d)", attempt)
                    if attempt == retries:
                        raise RuntimeError("VmClient call timed out after retries") from None
                except ConnectError as exc:
                    logger.warning(
                        "VmClient ConnectError %s %s (attempt %d)", exc.code, exc.message, attempt
                    )
                    if attempt == retries or exc.code not in _TRANSIENT_CODES:
                        raise
            time.sleep(2.0 ** attempt)
        raise RuntimeError("VmClient._call exhausted retries")  # unreachable

    def tree(self, root: str = "") -> dict[str, Any]:
        return self._call(lambda: self._vm.tree(TreeRequest(root=root)))

    def find(
        self, name: str, root: str = "/", kind: str = "all", limit: int = 10
    ) -> dict[str, Any]:
        type_map = {"all": 0, "files": 1, "dirs": 2}
        req = FindRequest(root=root, name=name, type=type_map[kind], limit=limit)
        return self._call(lambda: self._vm.find(req))

    def search(self, pattern: str, limit: int = 10, root: str = "/") -> dict[str, Any]:
        req = SearchRequest(root=root, pattern=pattern, limit=limit)
        return self._call(lambda: self._vm.search(req))

    def list(self, path: str = "/") -> dict[str, Any]:
        return self._call(lambda: self._vm.list(ListRequest(name=path)))

    def read(self, path: str) -> dict[str, Any]:
        return self._call(lambda: self._vm.read(ReadRequest(path=path)))

    def write(self, path: str, content: str) -> dict[str, Any]:
        return self._call(lambda: self._vm.write(WriteRequest(path=path, content=content)))

    def delete(self, path: str) -> dict[str, Any]:
        return self._call(lambda: self._vm.delete(DeleteRequest(path=path)))

    def mkdir(self, path: str) -> dict[str, Any]:
        return self._call(lambda: self._vm.mk_dir(MkDirRequest(path=path)))

    def move(self, from_name: str, to_name: str) -> dict[str, Any]:
        req = MoveRequest(from_name=from_name, to_name=to_name)
        return self._call(lambda: self._vm.move(req))

    def context(self) -> dict[str, Any]:
        return self._call(lambda: self._vm.context(ContextRequest()))

    def answer(
        self, message: str, outcome: str, refs: list[str] | None = None
    ) -> dict[str, Any]:
        req = AnswerRequest(
            message=message,
            outcome=_OUTCOME_BY_NAME[outcome],
            refs=refs or [],
        )
        return self._call(lambda: self._vm.answer(req))
