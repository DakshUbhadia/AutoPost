"""Durable training evidence, audit-only consensus flags and atomic sampling."""

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from autopost.contracts import ArtifactComponent, ArtifactManifest
from autopost.contracts.serialization import canonical_bytes, content_hash
from autopost.contracts.signals import (
    CurriculumConfig,
    LabelReview,
    PassRateObservation,
    SliceReservation,
)
from autopost.curriculum import choose
from autopost.interfaces import band_for, posterior_pass_rate

from .datasets import DatasetRepository, training_rows
from .db import connect, migrate_db
from .errors import ConflictError, StoreError


def task_digest(task):
    return hashlib.sha256(task.to_bytes()).hexdigest()


class TrainingSignalRepository:
    def __init__(self, path: str | Path, artifacts):
        self.path, self.artifacts = Path(path), artifacts
        migrate_db(self.path)
        self.datasets = DatasetRepository(self.path, artifacts)

    @contextmanager
    def _connection(self, *, write=False):
        db = connect(self.path)
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def retain_evidence(self, payload: dict):
        """Publish raw text before metadata; an interrupted write is effect-free."""
        blob = canonical_bytes(payload)
        digest = hashlib.sha256(blob).hexdigest()
        manifest = ArtifactManifest.create(
            artifact_id="signal-" + digest,
            kind="evidence",
            producer_job_id="signal-" + digest,
            components=(
                ArtifactComponent(
                    name="evidence.json",
                    sha256=digest,
                    size_bytes=len(blob),
                    media_type="application/json",
                ),
            ),
        )
        staged = self.artifacts.stage(manifest.artifact.artifact_id)
        self.artifacts.write(staged, "evidence.json", blob)
        self.artifacts.finalize(staged, manifest)
        return manifest.artifact

    def record_group(
        self, dataset, task, group_id, pairs, *, fixture_only=False, config=None
    ):
        """Each pair is a raw Completion and its training-verifier Verdict."""
        config = config or CurriculumConfig()
        self.datasets.assert_training_tasks(dataset, (task,), fixture_only=fixture_only)
        pairs = tuple(pairs)
        if not pairs:
            raise ValueError("an observation needs samples")
        completions, verdicts = zip(*pairs)
        checker = verdicts[0].checker_id
        if (
            len({c.completion_id for c in completions}) != len(completions)
            or len({(c.state_id, c.template_id, c.seed) for c in completions}) != 1
            or len({c.sample_idx for c in completions}) != len(completions)
            or any(c.task_id != task.task_id for c in completions)
            or any(v.checker_id != checker for v in verdicts)
            or checker.startswith("exact_numeric")
        ):
            raise ValueError(
                "group must contain unique training samples from one checker/state/template/seed"
            )
        for completion, verdict in pairs:
            if (
                verdict.correct != (verdict.status == "ok")
                or (completion.truncated and verdict.status != "truncated")
                or verdict.status
                not in {
                    "ok",
                    "wrong",
                    "no_answer",
                    "multiple_answers",
                    "truncated",
                    "parse_error",
                    "timeout",
                }
            ):
                raise ValueError("inconsistent train-verifier verdict")
        evidence = self.retain_evidence(
            {
                "schema": "autopost.training-group/1",
                "group_id": group_id,
                "task": task.to_dict(),
                "samples": [
                    {"completion": asdict(c), "verdict": asdict(v)} for c, v in pairs
                ],
            }
        )
        observation = PassRateObservation(
            group_id=group_id,
            task_id=task.task_id,
            task_digest=task_digest(task),
            state_id=completions[0].state_id,
            checker_id=checker,
            n=len(pairs),
            n_correct=sum(v.correct for v in verdicts),
            wrong_answers=dict(
                Counter(
                    v.extracted for v in verdicts if v.status == "wrong" and v.extracted
                )
            ),
            evidence=evidence,
        )
        blob = observation.to_bytes()
        with self._connection(write=True) as db:
            # Revalidate isolation inside the same transaction as the write.
            self._task(db, dataset, task, fixture_only)
            old = db.execute(
                "SELECT record FROM ap_passrate_groups WHERE group_id=?", (group_id,)
            ).fetchone()
            if old:
                if old[0] != blob:
                    raise ConflictError(
                        "logical rollout group already has different evidence"
                    )
                return PassRateObservation.from_bytes(old[0])
            db.execute(
                "INSERT INTO ap_passrate_groups VALUES (?,?,?,?,?)",
                (group_id, task.task_id, observation.task_digest, checker, blob),
            )
            if any(
                db.execute(
                    "SELECT 1 FROM ap_training_samples WHERE completion_id=?",
                    (c.completion_id,),
                ).fetchone()
                for c in completions
            ):
                raise ConflictError(
                    "completion already counted in a different rollout group"
                )
            db.executemany(
                "INSERT INTO ap_training_samples VALUES (?,?)",
                [(c.completion_id, group_id) for c in completions],
            )
            observations = self._observations(db, task, ())
            wrong = Counter()
            for item in observations:
                for answer, count in item.wrong_answers.items():
                    if count / item.n >= config.consensus_share:
                        wrong[answer] += 1
            if any(count >= config.min_estimates for count in wrong.values()):
                self._event(
                    db,
                    "consensus-" + group_id,
                    task.task_id,
                    "suspected_label_error",
                    {
                        "groups": [o.group_id for o in observations],
                        "config": config.model_dump(),
                        "evidence": evidence.model_dump(),
                    },
                )
        return observation

    @staticmethod
    def _task(db, dataset, task, fixture_only, *, include_quarantined=False):
        tasks = training_rows(
            db,
            dataset,
            fixture_only=fixture_only,
            include_quarantined=include_quarantined,
        )
        if not any(candidate.to_bytes() == task.to_bytes() for candidate in tasks):
            raise StoreError("task is outside immutable training membership")

    @staticmethod
    def _event(db, event_id, task_id, kind, payload):
        blob = canonical_bytes(payload)
        old = db.execute(
            "SELECT record,kind,task_id FROM ap_training_audit WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if old:
            if tuple(old) != (blob, kind, task_id):
                raise ConflictError("training audit identity has different content")
            return
        db.execute(
            "INSERT INTO ap_training_audit(event_id,task_id,kind,record) VALUES (?,?,?,?)",
            (event_id, task_id, kind, blob),
        )

    @staticmethod
    def _observations(db, task, excluded):
        # A corrected immutable label is retired; its dependent observations
        # cannot be reused even if a later review calls the old label good.
        reviews = [
            LabelReview.from_bytes(row[0])
            for row in db.execute(
                "SELECT record FROM ap_label_reviews WHERE task_id=? AND task_digest=? ORDER BY sequence",
                (task.task_id, task_digest(task)),
            )
        ]
        if any(review.decision == "corrected" for review in reviews):
            return ()
        return tuple(
            PassRateObservation.from_bytes(row[0])
            for row in db.execute(
                "SELECT record FROM ap_passrate_groups WHERE task_id=? AND task_digest=? ORDER BY group_id",
                (task.task_id, task_digest(task)),
            )
            if PassRateObservation.from_bytes(row[0]).checker_id not in excluded
        )

    @staticmethod
    def _excluded(db, task):
        rows = db.execute(
            "SELECT record FROM ap_label_reviews WHERE task_id=? AND task_digest=? ORDER BY sequence",
            (task.task_id, task_digest(task)),
        ).fetchall()
        reviews = [LabelReview.from_bytes(row[0]) for row in rows]
        return bool(
            reviews
            and (
                reviews[-1].decision != "confirmed_good"
                or any(r.decision in {"corrected", "irreparable"} for r in reviews)
            )
        )

    def estimate(self, task, *, excluded_checkers=()):
        with self._connection() as db:
            observations = self._observations(db, task, excluded_checkers)
            return posterior_pass_rate(
                sum(o.n_correct for o in observations), sum(o.n for o in observations)
            )

    def review(self, dataset, task, review: LabelReview, *, fixture_only=False):
        review.to_bytes()
        if review.task_id != task.task_id or review.task_digest != task_digest(task):
            raise ValueError("review must bind the exact immutable task/label")
        self.datasets.manifest(dataset)
        self.artifacts.resolve(review.evidence)
        if review.checker_id.startswith("exact_numeric"):
            raise ValueError("measurement scorer cannot decide training eligibility")
        with self._connection(write=True) as db:
            self._task(db, dataset, task, fixture_only, include_quarantined=True)
            if review.method == "second_verifier" and any(
                o.checker_id == review.checker_id
                for o in self._observations(db, task, ())
            ):
                raise ValueError("second verifier must differ from the reward checker")
            old = db.execute(
                "SELECT record FROM ap_label_reviews WHERE review_id=?",
                (review.review_id,),
            ).fetchone()
            if old:
                if old[0] != review.to_bytes():
                    raise ConflictError(
                        "review identity already has different evidence"
                    )
                return
            db.execute(
                "INSERT INTO ap_label_reviews(review_id,task_id,task_digest,record) VALUES (?,?,?,?)",
                (review.review_id, task.task_id, task_digest(task), review.to_bytes()),
            )
            self._event(
                db,
                "review-" + review.review_id,
                task.task_id,
                review.decision,
                {"review": review.model_dump()},
            )

    def reserve_slice(
        self,
        dataset,
        slice_id,
        n_prompts,
        seed,
        config,
        *,
        excluded_checkers=(),
        fixture_only=False,
    ):
        self.datasets.manifest(dataset)
        band = band_for(config.group_size, config.min_signal_prob)
        excluded_checkers = tuple(sorted(set(excluded_checkers)))
        request = {
            "slice_id": slice_id,
            "dataset": dataset,
            "config_digest": content_hash(config),
            "seed": seed,
            "n_prompts": n_prompts,
            "excluded_checkers": excluded_checkers,
        }
        SliceReservation(
            **request, task_ids=()
        ).to_bytes()  # validate before side effects
        with self._connection(write=True) as db:
            tasks = training_rows(db, dataset, fixture_only=fixture_only)
            old = db.execute(
                "SELECT record FROM ap_curriculum_slices WHERE slice_id=?", (slice_id,)
            ).fetchone()
            if old:
                reservation = SliceReservation.from_bytes(old[0])
                if reservation != SliceReservation(
                    **request, task_ids=reservation.task_ids
                ):
                    raise ConflictError(
                        "slice identity already has different semantic inputs"
                    )
                lookup = {task.task_id: task for task in tasks}
                if any(key not in lookup for key in reservation.task_ids):
                    raise StoreError(
                        "reserved slice contains a subsequently excluded task"
                    )
                selected = tuple(lookup[key] for key in reservation.task_ids)
                if any(self._excluded(db, task) for task in selected):
                    raise StoreError(
                        "reserved slice contains a subsequently excluded task"
                    )
                return selected
            eligible = []
            for task in tasks:
                obs = self._observations(db, task, excluded_checkers)
                n = sum(o.n for o in obs)
                p = posterior_pass_rate(sum(o.n_correct for o in obs), n)
                if (
                    n >= config.estimate_k
                    and band[0] <= p <= band[1]
                    and not self._excluded(db, task)
                ):
                    eligible.append(task)
            exposures = {
                row[0]: row[1]
                for row in db.execute(
                    "SELECT cluster_id,count(*) FROM ap_cluster_exposures GROUP BY cluster_id"
                )
            }
            selected = choose(tuple(eligible), n_prompts, seed, config, exposures)
            reservation = SliceReservation(
                **request, task_ids=tuple(t.task_id for t in selected)
            )
            db.execute(
                "INSERT INTO ap_curriculum_slices VALUES (?,?)",
                (slice_id, reservation.to_bytes()),
            )
            db.executemany(
                "INSERT INTO ap_cluster_exposures VALUES (?,?,?)",
                [(slice_id, task.task_id, task.cluster_id) for task in selected],
            )
        return selected

    def promotion_refresh(
        self, dataset, promotion_id, config, *, excluded_checkers=(), fixture_only=False
    ):
        """Durable band snapshots flag persistence and identify edge refresh work."""
        self.datasets.manifest(dataset)
        band = band_for(config.group_size, config.min_signal_prob)
        edges = []
        with self._connection(write=True) as db:
            for task in training_rows(db, dataset, fixture_only=fixture_only):
                obs = self._observations(db, task, excluded_checkers)
                n = sum(o.n for o in obs)
                p = posterior_pass_rate(sum(o.n_correct for o in obs), n)
                eligible = n >= config.estimate_k and not self._excluded(db, task)
                payload = {
                    "promotion": promotion_id,
                    "config": config.model_dump(),
                    "excluded_checkers": sorted(excluded_checkers),
                    "p": p,
                    "in_band": eligible and band[0] <= p <= band[1],
                    "near_edge": eligible
                    and min(abs(p - edge) for edge in band) <= config.edge_margin,
                }
                event_id = "promotion-" + content_hash(
                    {"id": promotion_id, "task": task_digest(task)}
                )
                old = db.execute(
                    "SELECT record FROM ap_training_audit WHERE event_id=?", (event_id,)
                ).fetchone()
                if old:
                    payload = json.loads(
                        old[0]
                    )  # retry returns the original refresh plan
                    if payload["config"] != config.model_dump() or payload[
                        "excluded_checkers"
                    ] != sorted(excluded_checkers):
                        raise ConflictError("promotion refresh inputs changed")
                else:
                    self._event(db, event_id, task.task_id, "promotion_band", payload)
                if payload["near_edge"]:
                    edges.append(task)
                snapshots = [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT record FROM ap_training_audit WHERE task_id=? AND kind='promotion_band' ORDER BY sequence DESC LIMIT ?",
                        (task.task_id, config.persistent_promotions),
                    )
                ]
                if (
                    not old
                    and len(snapshots) == config.persistent_promotions
                    and all(s["in_band"] for s in snapshots)
                ):
                    self._event(
                        db,
                        "persistent-" + event_id,
                        task.task_id,
                        "persistent_in_band",
                        {"promotion": promotion_id, "snapshots": snapshots},
                    )
        return tuple(edges)

    def audit(self, task_id):
        with self._connection() as db:
            return tuple(
                {"kind": row[0], "payload": json.loads(row[1])}
                for row in db.execute(
                    "SELECT kind,record FROM ap_training_audit WHERE task_id=? ORDER BY sequence",
                    (task_id,),
                )
            )

    def exposures(self):
        with self._connection() as db:
            return {
                row[0]: row[1]
                for row in db.execute(
                    "SELECT cluster_id,count(*) FROM ap_cluster_exposures GROUP BY cluster_id"
                )
            }
