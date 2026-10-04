import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from llm_router.canary import CanaryDecision, canary_plans
from llm_router.registry import load_registry
from llm_router.rollout import deploy_command, main, next_step, rollback_command

PLANS = {plan.id: plan for plan in canary_plans(load_registry("config/registry.yaml"))}
MODEL, ADAPTER, POLICY = "model:deploy-0002", "adapter:claims-extraction-lora-next", "policy:v1"
HEALTHY = {"requests": 800, "errors": 1, "p95_latency_ms": 400, "quality": 0.9}
FAILING = {"requests": 800, "errors": 200, "p95_latency_ms": 400, "quality": 0.9}
EARLY = {"requests": 10, "errors": 0}
ROLLBACK = CanaryDecision(action="rollback", reasons=("error rate 0.250 exceeds 0.010",))


class Recorder:
    def __init__(self, result: int = 0) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.result = result

    def __call__(self, command: Sequence[str]) -> int:
        self.commands.append(tuple(command))
        return self.result


def observation(tmp_path: Path, values: dict[str, object]) -> str:
    path = tmp_path / f"observation-{len(list(tmp_path.iterdir()))}.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return str(path)


def test_a_deploy_is_atomic_so_failed_readiness_undoes_it() -> None:
    command = deploy_command(values=["serving.mode=ray"])

    assert command[:5] == ("helm", "upgrade", "--install", "llm-routing", "deploy/helm/llm-routing")
    assert "--atomic" in command
    assert command[command.index("--timeout") + 1] == "10m"
    assert command[-2:] == ("--set", "serving.mode=ray")


def test_a_failed_model_canary_rolls_the_release_back_to_its_recorded_target() -> None:
    step = next_step(PLANS[MODEL], ROLLBACK, release="prod", namespace="inference")

    assert step.command == rollback_command(release="prod", namespace="inference")
    assert step.command[:3] == ("helm", "rollback", "prod") and "--wait" in step.command
    assert step.rollback_to == "deploy-0001"
    assert "redeploy deploy-0001" in step.note


def test_a_failed_adapter_canary_needs_no_redeploy() -> None:
    step = next_step(PLANS[ADAPTER], ROLLBACK)

    assert step.command == ()
    assert "gateway withdraws a failing adapter itself" in step.note


def test_a_rollback_with_no_recorded_target_runs_nothing() -> None:
    step = next_step(PLANS[POLICY], ROLLBACK)

    assert step.rollback_to is None and step.command == ()
    assert "no command is run without a recorded target" in step.note


@pytest.mark.parametrize("action", ["promote", "hold"])
def test_only_a_rollback_decision_produces_a_command(action: str) -> None:
    step = next_step(PLANS[MODEL], CanaryDecision(action=action))

    assert step.command == () and step.note == ""


@pytest.mark.parametrize(
    ("values", "code", "action"),
    [(HEALTHY, 0, "promote"), (EARLY, 2, "hold"), (FAILING, 3, "rollback")],
)
def test_the_exit_code_tells_a_pipeline_what_was_decided(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    values: dict[str, object],
    code: int,
    action: str,
) -> None:
    runner = Recorder()

    result = main(
        ["evaluate", "--plan", MODEL, "--observation", observation(tmp_path, values)], runner
    )

    printed = json.loads(capsys.readouterr().out)
    assert result == code and printed["action"] == action
    # Without --execute nothing is ever run.
    assert runner.commands == [] and printed["executed"] is False


def test_execute_runs_the_rollback_and_reports_when_it_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "evaluate",
        "--plan",
        MODEL,
        "--observation",
        observation(tmp_path, FAILING),
        "--execute",
    ]
    succeeded, failed = Recorder(), Recorder(result=1)

    assert main(arguments, succeeded) == 3
    assert succeeded.commands == [rollback_command()]
    assert json.loads(capsys.readouterr().out)["succeeded"] is True

    assert main(arguments, failed) == 4
    assert json.loads(capsys.readouterr().out)["succeeded"] is False


def test_execute_runs_nothing_for_a_healthy_canary_or_an_adapter(tmp_path: Path) -> None:
    runner = Recorder()

    healthy = ["evaluate", "--plan", MODEL, "--observation", observation(tmp_path, HEALTHY)]
    adapter = ["evaluate", "--plan", ADAPTER, "--observation", observation(tmp_path, FAILING)]
    assert main([*healthy, "--execute"], runner) == 0
    assert main([*adapter, "--execute"], runner) == 3
    assert runner.commands == []


def test_a_baseline_keeps_a_shared_outage_from_being_blamed_on_the_canary(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(FAILING), encoding="utf-8")
    runner = Recorder()

    result = main(
        [
            "evaluate",
            "--plan",
            MODEL,
            "--observation",
            observation(tmp_path, FAILING),
            "--baseline",
            str(baseline),
            "--execute",
        ],
        runner,
    )

    assert result != 3 and runner.commands == []


def test_deploy_prints_its_command_and_runs_it_only_when_asked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = Recorder()

    assert main(["deploy", "--set", "serving.mode=ray"], runner) == 0
    assert runner.commands == []
    printed = json.loads(capsys.readouterr().out)
    assert "--atomic" in printed["command"] and printed["executed"] is False

    assert main(["deploy", "--execute"], runner) == 0
    assert runner.commands == [deploy_command()]


@pytest.mark.parametrize("arguments", [["evaluate"], ["evaluate", "--plan", "model:unknown"]])
def test_evaluate_says_what_it_is_missing(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(arguments, Recorder())
    assert raised.value.code == 2


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_helm_accepts_the_flags_the_commands_use() -> None:
    for command in (deploy_command(), rollback_command()):
        flags = [item for item in command if item.startswith("--")]
        described = subprocess.run(
            [*command[:2], "--help"], capture_output=True, text=True, check=True
        ).stdout
        for flag in flags:
            assert flag in described, f"{flag} is not a flag of {' '.join(command[:2])}"
