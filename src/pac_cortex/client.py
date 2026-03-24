"""BitGN API clients: harness lifecycle and per-task VM runtime."""

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
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

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

    def _to_dict(self, result: Any) -> dict[str, Any]:
        return MessageToDict(result) if result else {}

    def tree(self, root: str = "") -> dict[str, Any]:
        return self._to_dict(self._vm.tree(TreeRequest(root=root)))

    def find(
        self, name: str, root: str = "/", kind: str = "all", limit: int = 10
    ) -> dict[str, Any]:
        type_map = {"all": 0, "files": 1, "dirs": 2}
        return self._to_dict(
            self._vm.find(FindRequest(root=root, name=name, type=type_map[kind], limit=limit))
        )

    def search(self, pattern: str, limit: int = 10, root: str = "/") -> dict[str, Any]:
        return self._to_dict(
            self._vm.search(SearchRequest(root=root, pattern=pattern, limit=limit))
        )

    def list(self, path: str = "/") -> dict[str, Any]:
        return self._to_dict(self._vm.list(ListRequest(name=path)))

    def read(self, path: str) -> dict[str, Any]:
        return self._to_dict(self._vm.read(ReadRequest(path=path)))

    def write(self, path: str, content: str) -> dict[str, Any]:
        return self._to_dict(self._vm.write(WriteRequest(path=path, content=content)))

    def delete(self, path: str) -> dict[str, Any]:
        return self._to_dict(self._vm.delete(DeleteRequest(path=path)))

    def mkdir(self, path: str) -> dict[str, Any]:
        return self._to_dict(self._vm.mk_dir(MkDirRequest(path=path)))

    def move(self, from_name: str, to_name: str) -> dict[str, Any]:
        return self._to_dict(self._vm.move(MoveRequest(from_name=from_name, to_name=to_name)))

    def answer(
        self, message: str, outcome: str, refs: list[str] | None = None
    ) -> dict[str, Any]:
        return self._to_dict(
            self._vm.answer(
                AnswerRequest(
                    message=message,
                    outcome=_OUTCOME_BY_NAME[outcome],
                    refs=refs or [],
                )
            )
        )
