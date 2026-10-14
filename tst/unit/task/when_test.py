"""A step's ``when``: what it takes, how the run's seed decides it, and what a skipped step leaves behind."""

import pytest

from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.store import Filter
from agent_env.task import Task
from agent_env.task.store import TaskStepStatus, find_task_instance, get_task_store
from agent_env.task.task import substitute_seed_in_steps
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, evaluate_when


class _Step(TaskStep):
    """Records that it ran."""

    type = "when_test"
    entity_refs = ()
    ran: list[str] = []

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        _Step.ran.append(self.id)
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Step.type: _Step}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Step, "ran", [])


def _step(id, depends_on=None, when=None, fail_task_on_error=True):
    step = _Step(id=id, version=None, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
    step.when = when
    return step


def _task(*steps):
    return Task(id="t", version=None, steps=list(steps))


@pytest.mark.parametrize("when", [None, True, "true", "TRUE", " yes ", "on", "1"])
def test_these_run(when):
    assert evaluate_when(when) is True


@pytest.mark.parametrize("when", [False, "false", "False", "no", "off", "0", "", "  "])
def test_these_skip(when):
    assert evaluate_when(when) is False


def test_a_placeholder_the_seed_never_filled_is_an_error_naming_it():
    with pytest.raises(ValueError, match="still holds a placeholder"):
        evaluate_when("<needs_sketchup>")


def test_anything_else_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="is not a boolean"):
        evaluate_when("maybe")


def test_when_is_a_bool_or_a_string():
    with pytest.raises(ValueError, match="when is a bool or a string, not 3"):
        _Step(id="a", version=None, when=3)


@pytest.mark.asyncio
async def test_a_skipped_step_does_not_run_and_its_dependents_still_do(local_stores):
    task = _task(_step("a"), _step("b", ["a"], when="false"), _step("c", ["b"]))
    ctx = TaskStepContext()

    await task.run(context=ctx)

    assert _Step.ran == ["a", "c"]
    inst = find_task_instance(ctx.instance_id)
    assert {s.step_id: s.status for s in inst.completed_steps} == {
        "a": TaskStepStatus.SUCCESS, "b": TaskStepStatus.SKIPPED, "c": TaskStepStatus.SUCCESS}
    assert inst.status == "completed"
    assert "failed_steps" not in ctx.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "ran"), [("true", ["a", "b"]), ("false", ["b"])])
async def test_the_runs_seed_decides(local_stores, value, ran):
    task = _task(_step("a", when="<needs_sketchup>"), _step("b", ["a"]))
    ctx = TaskStepContext()
    ctx.metadata["seed"] = {"needs_sketchup": value}

    await task.run(context=ctx)

    assert _Step.ran == ran


@pytest.mark.asyncio
async def test_a_run_without_the_seed_value_fails_the_step_instead_of_guessing(local_stores):
    task = _task(_step("a", when="<needs_sketchup>"), _step("b", ["a"]))
    ctx = TaskStepContext()
    ctx.metadata["seed"] = {"brief": "unrelated"}

    with pytest.raises(ValueError, match="still holds a placeholder"):
        await task.run(context=ctx)

    assert _Step.ran == []


@pytest.mark.asyncio
async def test_a_tolerant_step_with_a_bad_when_fails_tolerantly(local_stores):
    task = _task(_step("a", when="maybe", fail_task_on_error=False), _step("b", ["a"]))
    ctx = TaskStepContext()

    await task.run(context=ctx)

    assert _Step.ran == ["b"]
    assert [f["step_id"] for f in ctx.metadata["failed_steps"]] == ["a"]


def test_when_round_trips_through_the_task_store_and_seed_substitution(local_stores):
    get_task_store().put_document(_task(_step("a", when="<needs_sketchup>"), _step("b")))

    stored = get_config().get_document_store().find_one("tasks", Filter.of(id="t"))
    assert [s.get("when") for s in stored["steps"]] == ["<needs_sketchup>", None]
    steps = Task.get("t").steps
    assert steps[0].when == "<needs_sketchup>"
    assert [s.when for s in substitute_seed_in_steps(steps, {"needs_sketchup": "no"})] == ["no", None]


def test_put_takes_when_on_a_step_whose_init_does_not(local_stores):
    saved = _Step.put(id="a", when="<needs_sketchup>")

    assert saved.when == "<needs_sketchup>"
    assert TaskStep.get("a").when == "<needs_sketchup>"
