"""Trainer gateways depend on a catalog contract, never on a storage backend."""

from typing import Protocol

from autopost.contracts.refs import DatasetRef
from autopost.contracts.tasks import Task
from autopost.interfaces import OfflineExample, assert_not_measurement


class TrainingCatalog(Protocol):
    def training_tasks(
        self, ref: DatasetRef, *, fixture_only: bool = False
    ) -> tuple[Task, ...]: ...
    def assert_training_tasks(
        self, ref: DatasetRef, tasks: tuple[Task, ...], *, fixture_only: bool = False
    ) -> None: ...


class TrainingDataAccess:
    def __init__(self, catalog: TrainingCatalog, *, fixture_only: bool = False):
        self.catalog = catalog
        self.fixture_only = fixture_only

    def on_policy(self, ref: DatasetRef) -> tuple[Task, ...]:
        tasks = self.catalog.training_tasks(ref, fixture_only=self.fixture_only)
        assert_not_measurement(tasks)
        return tasks

    def offline(
        self, ref: DatasetRef, examples: tuple[OfflineExample, ...]
    ) -> tuple[OfflineExample, ...]:
        tasks = tuple(example.task for example in examples)
        assert_not_measurement(tasks)
        self.catalog.assert_training_tasks(ref, tasks, fixture_only=self.fixture_only)
        if any(
            not example.provenance
            or not example.verdict.correct
            or example.verdict.status != "ok"
            for example in examples
        ):
            raise ValueError(
                "offline examples require correct independent train-verifier verdicts and provenance"
            )
        return examples
