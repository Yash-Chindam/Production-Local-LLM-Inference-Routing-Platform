"""Acting on a rollout: deploy atomically, and roll back when a canary fails.

Section 16 asks for automatic rollback after failed readiness or failed canary
criteria. Readiness is covered by deploying with Helm's ``--atomic``: a release
whose pods never become ready is rolled back by Helm itself. Canary criteria
are judged by ``llm_router.canary``; this module turns that verdict into the
command that carries it out, so the model and policy tracks roll back without
a person reading a decision and typing one.

Adapters are not handled here. The gateway withdraws a failing adapter canary
on its own, with no redeploy.
"""

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from llm_router.canary import (
    CanaryDecision,
    CanaryObservation,
    CanaryPlan,
    CanaryTrack,
    canary_plans,
    evaluate,
)

DEFAULT_RELEASE = "llm-routing"
DEFAULT_NAMESPACE = "llm-routing"
DEFAULT_CHART = "deploy/helm/llm-routing"
DEFAULT_TIMEOUT = "10m"
EXIT_CODES = {"promote": 0, "hold": 2, "rollback": 3}
# The decision was to roll back and the rollback itself did not succeed.
EXIT_ROLLBACK_FAILED = 4

Runner = Callable[[Sequence[str]], int]


class RolloutStep(BaseModel):
    """What a canary verdict means for the release, and the command that does it."""

    plan: str
    action: str
    reasons: tuple[str, ...]
    rollback_to: str | None
    command: tuple[str, ...] = ()
    note: str = ""


def deploy_command(
    *,
    release: str = DEFAULT_RELEASE,
    namespace: str = DEFAULT_NAMESPACE,
    chart: str = DEFAULT_CHART,
    values: Sequence[str] = (),
    timeout: str = DEFAULT_TIMEOUT,
) -> tuple[str, ...]:
    """Install or upgrade the release so that failed readiness undoes it.

    ``--atomic`` waits for every workload to become ready and, if one does not
    within the timeout, restores the previous release.
    """

    command = [
        "helm",
        "upgrade",
        "--install",
        release,
        chart,
        "--namespace",
        namespace,
        "--create-namespace",
        "--atomic",
        "--timeout",
        timeout,
    ]
    for value in values:
        command += ["--set", value]
    return tuple(command)


def rollback_command(
    *,
    release: str = DEFAULT_RELEASE,
    namespace: str = DEFAULT_NAMESPACE,
    timeout: str = DEFAULT_TIMEOUT,
) -> tuple[str, ...]:
    """Restore the release that was live before the current one."""

    return ("helm", "rollback", release, "--namespace", namespace, "--wait", "--timeout", timeout)


def next_step(
    plan: CanaryPlan,
    decision: CanaryDecision,
    *,
    release: str = DEFAULT_RELEASE,
    namespace: str = DEFAULT_NAMESPACE,
) -> RolloutStep:
    """Translate a verdict on one plan into the step that carries it out."""

    step = RolloutStep(
        plan=plan.id,
        action=decision.action,
        reasons=decision.reasons,
        rollback_to=plan.rollback_to,
    )
    if decision.action != "rollback":
        return step
    if plan.track is CanaryTrack.ADAPTER:
        return step.model_copy(
            update={"note": "the gateway withdraws a failing adapter itself; nothing to redeploy"}
        )
    if plan.rollback_to is None:
        # With nothing recorded to return to, a rollback would restore
        # whatever Helm happens to hold. Stopping is the only safe step.
        return step.model_copy(
            update={"note": f"{plan.rollback_action}; no command is run without a recorded target"}
        )
    return step.model_copy(
        update={
            "command": rollback_command(release=release, namespace=namespace),
            "note": plan.rollback_action,
        }
    )


def run_command(command: Sequence[str]) -> int:  # pragma: no cover - runs helm
    return subprocess.run(list(command), check=False).returncode


def main(argv: Sequence[str] | None = None, runner: Runner = run_command) -> int:
    """Deploy the release, or act on a canary observation.

    ``deploy`` prints the atomic install command. ``evaluate`` judges one plan
    against an observation and prints the step that follows. Either runs its
    command only with --execute; without it nothing is changed.

    ``evaluate`` exits 0 to promote, 2 to hold, 3 after a rollback decision,
    and 4 if the rollback command itself failed.
    """

    import argparse

    from llm_router.registry import load_registry

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["deploy", "evaluate"])
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--chart", default=DEFAULT_CHART)
    parser.add_argument("--set", action="append", default=[], dest="values")
    parser.add_argument("--catalog", default="config/registry.yaml")
    parser.add_argument("--plan", help="plan id to evaluate, for example model:deploy-0002")
    parser.add_argument("--observation", help="JSON file holding the observed canary metrics")
    parser.add_argument("--baseline", help="JSON file holding the stable baseline metrics")
    parser.add_argument("--execute", action="store_true", help="run the command, not just print it")
    arguments = parser.parse_args(argv)

    if arguments.command == "deploy":
        command = deploy_command(
            release=arguments.release,
            namespace=arguments.namespace,
            chart=arguments.chart,
            values=arguments.values,
        )
        print(json.dumps({"command": list(command), "executed": arguments.execute}, indent=2))
        return runner(command) if arguments.execute else 0

    plans = {plan.id: plan for plan in canary_plans(load_registry(arguments.catalog))}
    plan = plans.get(arguments.plan or "")
    if plan is None or arguments.observation is None:
        parser.error(f"evaluate needs --observation and --plan, one of: {', '.join(plans)}")

    def read(path: str) -> CanaryObservation:
        return CanaryObservation.model_validate_json(Path(path).read_text(encoding="utf-8"))

    decision = evaluate(
        plan.criteria,
        read(arguments.observation),
        read(arguments.baseline) if arguments.baseline else None,
    )
    step = next_step(plan, decision, release=arguments.release, namespace=arguments.namespace)
    executed = bool(step.command) and arguments.execute
    failed = executed and runner(step.command) != 0
    print(
        json.dumps(
            {**step.model_dump(mode="json"), "executed": executed, "succeeded": not failed},
            indent=2,
        )
    )
    return EXIT_ROLLBACK_FAILED if failed else EXIT_CODES[decision.action]


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
