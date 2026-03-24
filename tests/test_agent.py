from unittest.mock import MagicMock

from pac_cortex.agent import NextStep, ReportTaskCompletion, ReqTree, solve_task


def test_immediate_report_completion(
    mock_vm_client: MagicMock, mock_llm_client: MagicMock
) -> None:
    """Agent calls report_completion immediately → single LLM call, single vm.answer."""
    mock_llm_client.parse_step.return_value = NextStep(
        current_state="done",
        plan_remaining_steps_brief=["report"],
        task_completed=True,
        function=ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["done"],
            message="Task complete",
            grounding_refs=[],
            outcome="OUTCOME_OK",
        ),
    )

    solve_task("capture foo.md", mock_vm_client, mock_llm_client)

    assert mock_llm_client.parse_step.call_count == 1
    assert mock_vm_client.answer.call_count == 1
    mock_vm_client.answer.assert_called_once_with(
        message="Task complete",
        outcome="OUTCOME_OK",
        refs=[],
    )


def test_stagnation_triggers_err_internal(
    mock_vm_client: MagicMock, mock_llm_client: MagicMock
) -> None:
    """Agent repeating the same tool _MAX_STAGNATION times triggers OUTCOME_ERR_INTERNAL."""
    mock_llm_client.parse_step.return_value = NextStep(
        current_state="exploring",
        plan_remaining_steps_brief=["tree"],
        task_completed=False,
        function=ReqTree(tool="tree", root=""),
    )

    solve_task("find something", mock_vm_client, mock_llm_client)

    # Stagnation fires after 3 identical calls; dispatch runs on steps 1 and 2 only
    assert mock_llm_client.parse_step.call_count == 3
    assert mock_vm_client.tree.call_count == 2
    assert mock_vm_client.answer.call_count == 1
    mock_vm_client.answer.assert_called_once_with(
        message="Agent stuck in repeated tool loop",
        outcome="OUTCOME_ERR_INTERNAL",
    )
