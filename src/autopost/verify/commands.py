"""CLI adapters for auditable training verification and curriculum selection."""

import json
from dataclasses import asdict
from pathlib import Path

from autopost.config import load_config
from autopost.contracts.refs import DatasetRef
from autopost.contracts.signals import CurriculumConfig, LabelReview
from autopost.curriculum import CurriculumSampler
from autopost.interfaces import BinaryReward, Completion
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.signals import TrainingSignalRepository


def training_command(args):
    root = Path(args.root)
    artifacts = LocalArtifactStore(root / "artifacts")
    repo = TrainingSignalRepository(root / "datasets.sqlite", artifacts)
    dataset = DatasetRef.from_bytes(Path(args.dataset_ref).read_bytes())
    experiment = load_config(args.base_yaml)["experiment"]
    config = CurriculumConfig.from_experiment(experiment)
    if args.action == "sample":
        sampler = CurriculumSampler(
            repo,
            dataset,
            config,
            fixture_only=args.fixture_only,
            excluded_checkers=tuple(args.exclude_checker),
        )
        tasks = sampler.reserve(args.slice_id, args.n_prompts, args.seed)
        return {
            "schema": "autopost.training-slice/1",
            "slice_id": args.slice_id,
            "task_ids": [task.task_id for task in tasks],
            "exposures": repo.exposures(),
        }
    if args.action == "refresh":
        tasks = repo.promotion_refresh(
            dataset,
            args.promotion_id,
            config,
            excluded_checkers=tuple(args.exclude_checker),
            fixture_only=args.fixture_only,
        )
        return {
            "schema": "autopost.training-refresh/1",
            "promotion_id": args.promotion_id,
            "task_ids": [task.task_id for task in tasks],
        }
    if args.action == "review":
        review = LabelReview.from_bytes(Path(args.review).read_bytes())
        task = next(
            (t for t in repo.datasets.inspect(dataset) if t.task_id == review.task_id),
            None,
        )
        if task is None:
            raise ValueError("review task is outside the selected dataset")
        repo.review(dataset, task, review, fixture_only=args.fixture_only)
        return {"review_id": review.review_id, "decision": review.decision}
    from autopost.verify.train import MathVerifyPool

    task = next(
        (
            t
            for t in repo.datasets.training_tasks(
                dataset, fixture_only=args.fixture_only
            )
            if t.task_id == args.task_id
        ),
        None,
    )
    if task is None:
        raise ValueError("verification task is outside eligible training membership")
    reward_config = experiment["reward"]
    reward = BinaryReward(
        args.max_new_tokens,
        reward_config["overlong_window"],
        reward_config["overlong_max_penalty"],
    )
    completions = tuple(
        Completion(**item)
        for item in json.loads(Path(args.completions).read_text(encoding="utf-8"))
    )
    verifier = MathVerifyPool(workers=args.workers, timeout_seconds=args.timeout)
    verdicts = verifier.check_many((task, completion) for completion in completions)
    observation = repo.record_group(
        dataset,
        task,
        args.group_id,
        zip(completions, verdicts),
        fixture_only=args.fixture_only,
        config=config,
    )
    return {
        "schema": "autopost.training-verification/1",
        "observation": observation.model_dump(mode="json"),
        "reward_id": reward.reward_id,
        "verdicts": [asdict(verdict) for verdict in verdicts],
        "rewards": [asdict(reward(task, c, v)) for c, v in zip(completions, verdicts)],
    }
