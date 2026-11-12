import json
from dataclasses import replace
from fractions import Fraction

import pytest

from autopost.contracts.datasets import BuildConfig, DatasetManifest, SourceSpec
from autopost.contracts.tasks import MEASUREMENT_SPLITS, SEALED_SPLITS, Task
from autopost.data.builder import build
from autopost.data.dedup import EmbeddingTable, cluster, minhash, shingles
from autopost.data.fixtures import development_inputs
from autopost.data.normalize import normalize, problem_key, type_answer
from autopost.data.sources import SourceRow, SourceSnapshot, read_jsonl


@pytest.fixture(scope="module")
def built_data():
    sources, embeddings, config, programs = development_inputs()
    return (
        sources,
        embeddings,
        config,
        programs,
        build(sources, embeddings, config, programs=programs),
    )


@pytest.mark.parametrize(
    "raw,answer,kind",
    [
        (" -12 ", "-12", "int"),
        ("6/8", "3/4", "rational"),
        (r"\frac{2}{6}", "1/3", "rational"),
        ("$0.125$", "1/8", "decimal"),
        ("½", "1/2", "rational"),
        ("sqrt(2)", "sqrt(2)", "expr"),
        ("[1,2]", "[1,2]", "interval"),
        ("(1,2)", "(1,2)", "tuple"),
    ],
)
def test_exact_answer_typing_and_symbolic_training(raw, answer, kind):
    assert type_answer(raw) == (answer, kind)


@pytest.mark.parametrize("raw", ["1/0", "", "yes", "false", "NaN", "Infinity"])
def test_invalid_or_low_information_labels_are_rejected(raw):
    with pytest.raises(ValueError):
        type_answer(raw)


def test_development_splits_are_disjoint_numeric_and_rejections_are_complete(
    built_data,
):
    *_, result = built_data
    by_cluster = {}
    assert {bundle.manifest.split for bundle in result.bundles} == {
        "train",
        "dev",
        "pilot",
        "screen",
        "gate_a",
        "gate_a_reservoir",
        "gate_b",
        "gate_b_reservoir",
        "transfer_guard",
        "hidden_dev",
    }
    for bundle in result.bundles:
        assert bundle.manifest.split not in SEALED_SPLITS
        assert DatasetManifest.from_bytes(bundle.manifest.to_bytes()) == bundle.manifest
        for task in bundle.tasks:
            assert Task.from_bytes(task.to_bytes()) == task
            assert task.source_ver != "legacy-unversioned"
            assert by_cluster.setdefault(task.cluster_id, task.split) == task.split
            if task.split in MEASUREMENT_SPLITS:
                assert task.measurable and isinstance(Fraction(task.answer), Fraction)
            if task.split in {"train", "dev"}:
                assert task.source_role == "train"
    reasons = {row["reason"] for row in result.report["rejections"]}
    assert {
        "invalid_numeric_answer",
        "yes_no_answer",
        "multiple_choice",
        "empty_or_oversized_problem",
        "benchmark_overlap",
        "benchmark_reference_only",
        "transfer_source_lineage",
        "duplicate_in_cluster",
    } <= reasons
    assert {edge["method"] for edge in result.report["dedup_edges"]} >= {
        "exact",
        "minhash",
        "embedding",
    }
    assert all(
        row["key"] and row["stage"] and row["reason"]
        for row in result.report["rejections"]
    )
    symbolic = [
        task
        for bundle in result.bundles
        for task in bundle.tasks
        if not task.measurable
    ]
    assert symbolic and all(task.split in {"train", "dev"} for task in symbolic)


def test_same_inputs_and_seed_produce_identical_ids_manifests_and_reports(built_data):
    sources, embeddings, config, programs, first = built_data
    reordered = tuple(
        replace(source, rows=tuple(reversed(source.rows)))
        for source in reversed(sources)
    )
    second = build(reordered, embeddings, config, programs=tuple(reversed(programs)))
    assert first.build_id == second.build_id
    assert first.report == second.report
    assert [bundle.manifest.to_bytes() for bundle in first.bundles] == [
        bundle.manifest.to_bytes() for bundle in second.bundles
    ]
    assert [task.to_bytes() for bundle in first.bundles for task in bundle.tasks] == [
        task.to_bytes() for bundle in second.bundles for task in bundle.tasks
    ]
    changed = build(
        sources, embeddings, config.model_copy(update={"seed": 7}), programs=programs
    )
    assert changed.build_id != first.build_id
    first_ids = {
        task.task_id
        for bundle in first.bundles
        for task in bundle.tasks
        if task.source_role != "hidden"
    }
    second_ids = {
        task.task_id
        for bundle in changed.bundles
        for task in bundle.tasks
        if task.source_role != "hidden"
    }
    assert first_ids == second_ids


@pytest.mark.parametrize(
    "split",
    ["hidden_audit", "hidden_sealed", "transfer_sealed", "hidden_dev", "unknown"],
)
def test_research_or_item_level_hidden_split_requests_fail_closed(split):
    with pytest.raises(ValueError):
        BuildConfig(split_clusters={split: 1})


def test_missing_unpinned_zero_and_inconsistent_embeddings_are_not_silently_skipped(
    built_data,
):
    sources, embeddings, config, programs, _ = built_data
    for vectors in (
        {},
        {key: (0.0,) for key in embeddings.vectors},
        {key: (float("nan"),) for key in embeddings.vectors},
    ):
        with pytest.raises(ValueError):
            build(
                sources, replace(embeddings, vectors=vectors), config, programs=programs
            )
    with pytest.raises(ValueError, match="pinned"):
        replace(embeddings, fixture_only=False).validate(
            tuple(
                normalize(source, row) for source in sources for row in source.rows[:1]
            )
        )


def test_lineage_and_embedding_assistance_cluster_paraphrases_transitively():
    spec = SourceSpec(
        source_id="source",
        revision="fixture-v1",
        role="train",
        family_id="family",
        license_id="CC0",
        fixture_only=True,
    )
    source = SourceSnapshot(
        spec,
        (
            SourceRow(
                "one",
                "Find the number of apples in a box.",
                "12",
                "count",
                ("original-item",),
            ),
            SourceRow(
                "two",
                "Count fruit stored in this container.",
                "12",
                "count",
                ("original-item",),
            ),
            SourceRow(
                "three",
                "Compute the cardinality of the contained fruit.",
                "12",
                "count",
            ),
        ),
    )
    rows = tuple(normalize(source, row) for row in source.rows)
    vectors = {
        problem_key(rows[0].problem): (0.0, 1.0),
        problem_key(rows[1].problem): (1.0, 0.0),
        problem_key(rows[2].problem): (1.0, 0.0),
    }
    groups, edges = cluster(
        rows,
        EmbeddingTable("fixture", "fixture-v1", vectors, True),
        BuildConfig(split_clusters={}),
    )
    assert len(groups) == 1 and len(groups[0]) == 3
    assert {edge["method"] for edge in edges} == {"lineage", "embedding"}
    assert minhash(shingles(rows[0].problem, 5), 64, 0) == minhash(
        shingles(rows[0].problem, 5), 64, 0
    )


def test_jsonl_source_adapter_requires_pins_and_preserves_provenance(tmp_path):
    spec = SourceSpec(
        source_id="curated",
        revision="a" * 40,
        role="train",
        family_id="family",
        license_id="CC0",
    )
    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps({"source": spec.model_dump()})
        + "\n"
        + json.dumps(
            {
                "source_item_id": "one",
                "problem": r"\( 6+6 \)",
                "answer": "12",
                "skill": "sum",
                "lineage_keys": ["upstream-item"],
            }
        ),
        encoding="utf-8",
    )
    snapshot = read_jsonl(path)
    normalized = normalize(snapshot, snapshot.rows[0])
    assert (
        normalized.source.spec.revision == "a" * 40
        and "upstream-item" in normalized.lineage
    )
    with pytest.raises(ValueError):
        SourceSpec(
            source_id="bad",
            revision="main",
            role="train",
            family_id="f",
            license_id="CC0",
        )
    with pytest.raises(ValueError):
        SourceRow("x", "problem", "12", "math", label_origin="model_generated")


def test_pinned_label_conflicts_and_invalid_benchmark_fail_closed(built_data):
    sources, embeddings, config, _, _ = built_data
    source = sources[0]
    duplicate = replace(source.rows[3], source_item_id="conflicting-copy", answer="73")
    bad_source = replace(source, rows=(*source.rows, duplicate))
    lowered = config.model_copy(
        update={"split_clusters": {"train": 1, "transfer_guard": 1}}
    )
    result = build((bad_source, *sources[1:]), embeddings, lowered)
    assert "conflicting_reference_labels" in {
        row["reason"] for row in result.report["rejections"]
    }
    benchmark = replace(sources[-1], rows=(replace(sources[-1].rows[0], problem=""),))
    with pytest.raises(ValueError, match="benchmark reference"):
        build((*sources[:-1], benchmark), embeddings, config)


def test_task_measurable_is_derived_and_numeric_types_are_enforced():
    task = Task(
        "task",
        "cluster",
        "source",
        "skill",
        r"\(1+2\)",
        "3",
        "int",
        True,
        "train",
        source_ver="a" * 40,
    )
    assert Task.from_bytes(task.to_bytes()) == task
    value = task.to_dict()
    value["measurable"] = False
    with pytest.raises(ValueError, match="measurable"):
        Task.from_dict(value)
    with pytest.raises(ValueError):
        replace(task, answer="not numeric")
    with pytest.raises(ValueError):
        replace(task, answer_type="expr", answer="x", split="gate_a")
    with pytest.raises(ValueError):
        replace(task, answer_type="decimal", answer="1/3")


@pytest.mark.parametrize(
    "adapter,row,expected",
    [
        (
            "gsm8k",
            {
                "id": "g1",
                "question": "How much is six plus six?",
                "answer": "Add six to six.\n#### 12",
            },
            "12",
        ),
        (
            "math",
            {
                "id": "m1",
                "problem": "Find a quarter.",
                "solution": r"The result is \boxed{\frac{1}{4}}.",
                "type": "Algebra",
            },
            "1/4",
        ),
    ],
)
def test_curated_dataset_adapters_preserve_original_labels_and_solutions(
    tmp_path, adapter, row, expected
):
    spec = SourceSpec(
        source_id=adapter,
        revision="b" * 40,
        role="train",
        family_id=adapter,
        license_id="declared",
        adapter=adapter,
    )
    path = tmp_path / "snapshot.jsonl"
    path.write_text(
        json.dumps({"source": spec.model_dump()}) + "\n" + json.dumps(row),
        encoding="utf-8",
    )
    snapshot = read_jsonl(path)
    normalized = normalize(snapshot, snapshot.rows[0])
    assert normalized.answer == expected and snapshot.rows[0].source_payload == row
