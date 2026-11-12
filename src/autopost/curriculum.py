"""Deterministic training-only policy; persistence is accessed via a protocol."""

from collections import Counter
from math import floor
from random import Random
from typing import Protocol

from autopost.contracts.refs import DatasetRef
from autopost.contracts.signals import CurriculumConfig
from autopost.contracts.tasks import Task
from autopost.interfaces import PromptSampler, band_for


class PoolExhaustedError(RuntimeError):
    """No complete slice fits the frozen constraints; never relax silently."""


def choose(
    tasks: tuple[Task, ...],
    n: int,
    seed: int,
    config: CurriculumConfig,
    exposures: dict[str, int],
) -> tuple[Task, ...]:
    if type(n) is not int or n < 1:
        raise ValueError("n_prompts must be positive")
    candidates = sorted(tasks, key=lambda task: task.task_id)
    Random(seed).shuffle(candidates)
    quota = floor(n * config.max_share)
    low_cap = floor(n * config.low_entropy_max_share)
    sources, skills, clusters = Counter(), Counter(), Counter(exposures)
    selected = []
    visits = 0

    def search(start, low):
        nonlocal visits
        visits += 1
        if visits > 100_000:
            raise PoolExhaustedError(
                "quota search budget exhausted; no exposures committed"
            )
        if len(selected) == n:
            return True
        if len(candidates) - start < n - len(selected):
            return False
        for index in range(start, len(candidates)):
            task = candidates[index]
            if (
                sources[task.source_id] >= quota
                or skills[task.skill] >= quota
                or clusters[task.cluster_id] >= config.cluster_exposure_cap
                or low + task.low_entropy > low_cap
            ):
                continue
            selected.append(task)
            sources[task.source_id] += 1
            skills[task.skill] += 1
            clusters[task.cluster_id] += 1
            if search(index + 1, low + task.low_entropy):
                return True
            selected.pop()
            sources[task.source_id] -= 1
            skills[task.skill] -= 1
            clusters[task.cluster_id] -= 1
        return False

    if not search(0, 0):
        raise PoolExhaustedError(
            "training pool cannot satisfy band, quotas and global caps"
        )
    return tuple(selected)


class CurriculumStore(Protocol):
    def reserve_slice(
        self,
        dataset: DatasetRef,
        slice_id: str,
        n_prompts: int,
        seed: int,
        config: CurriculumConfig,
        *,
        excluded_checkers: tuple[str, ...],
        fixture_only: bool,
    ) -> tuple[Task, ...]: ...


class CurriculumSampler(PromptSampler):
    """Call reserve with a durable logical slice ID for retry-safe accounting."""

    def __init__(
        self,
        store: CurriculumStore,
        dataset: DatasetRef,
        config: CurriculumConfig,
        *,
        fixture_only=False,
        excluded_checkers: tuple[str, ...] = (),
    ):
        band_for(config.group_size, config.min_signal_prob)
        self.store, self.dataset, self.config = store, dataset, config
        self.fixture_only, self.excluded_checkers = fixture_only, excluded_checkers

    def reserve(self, slice_id: str, n_prompts: int, seed: int) -> tuple[Task, ...]:
        return self.store.reserve_slice(
            self.dataset,
            slice_id,
            n_prompts,
            seed,
            self.config,
            excluded_checkers=self.excluded_checkers,
            fixture_only=self.fixture_only,
        )

    def next_slice(self, n_prompts: int, seed: int) -> list[Task]:
        from autopost.contracts.serialization import content_hash

        identity = content_hash(
            {
                "dataset": self.dataset.model_dump(),
                "config": self.config.model_dump(),
                "n": n_prompts,
                "seed": seed,
                "excluded": sorted(self.excluded_checkers),
            }
        )
        return list(self.reserve("slice-" + identity, n_prompts, seed))
