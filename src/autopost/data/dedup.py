"""Deterministic MinHash, cosine assistance and transitive source lineage."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from autopost.contracts.datasets import BuildConfig
from autopost.contracts.serialization import content_hash

from .normalize import NormalizedRow, normalize_text, problem_key


@dataclass(frozen=True)
class EmbeddingTable:
    model_id: str
    revision: str
    vectors: dict[str, tuple[float, ...]]
    fixture_only: bool = False

    def __post_init__(self):
        if (
            type(self.fixture_only) is not bool
            or type(self.model_id) is not str
            or not self.model_id
            or type(self.revision) is not str
        ):
            raise ValueError(
                "embedding identity and fixture provenance must be explicit"
            )

    def validate(self, rows: tuple[NormalizedRow, ...]):
        if not self.fixture_only and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("embedding model requires a pinned 40-hex revision")
        dimensions = set()
        for row in rows:
            vector = self.vectors.get(problem_key(row.problem))
            if vector is None:
                raise ValueError(
                    "missing precomputed embedding; refusing partial deduplication"
                )
            if not vector or any(
                type(value) not in (float, int) or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError("embedding vectors must be finite numeric arrays")
            norm = math.hypot(*vector)
            if norm == 0 or not math.isfinite(norm):
                raise ValueError("zero or nonfinite embedding norm")
            dimensions.add(len(vector))
        if len(dimensions) > 1:
            raise ValueError("inconsistent embedding dimensions")

    @property
    def digest(self):
        return content_hash(
            {
                "model_id": self.model_id,
                "revision": self.revision,
                "fixture_only": self.fixture_only,
                "vectors": self.vectors,
            }
        )


def shingles(text: str, ngram: int) -> frozenset[str]:
    tokens = re.findall(r"\w+|[^\w\s]", normalize_text(text).casefold())
    return frozenset(
        " ".join(tokens[i : i + ngram]) for i in range(max(1, len(tokens) - ngram + 1))
    )


def minhash(tokens: frozenset[str], permutations: int, seed: int) -> tuple[int, ...]:
    return tuple(
        min(
            int.from_bytes(
                hashlib.sha256(f"{seed}:{index}:{token}".encode()).digest()[:8], "big"
            )
            for token in tokens
        )
        for index in range(permutations)
    )


def cluster(
    rows: tuple[NormalizedRow, ...], embeddings: EmbeddingTable, config: BuildConfig
):
    rows = tuple(sorted(rows, key=lambda row: row.prompt_id))
    embeddings.validate(rows)
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    tokens = [shingles(row.problem, config.minhash_ngram) for row in rows]
    signatures = [
        minhash(value, config.minhash_permutations, config.seed) for value in tokens
    ]
    edges = []
    # Reference implementation deliberately checks all pairs. No approximate
    # candidate index may silently skip embedding or lineage comparisons.
    for i, left in enumerate(rows):
        for j in range(i):
            right = rows[j]
            reason, score = None, 1.0
            if problem_key(left.problem) == problem_key(right.problem):
                reason = "exact"
            elif set(left.lineage) & set(right.lineage):
                reason = "lineage"
            else:
                estimate = (
                    sum(
                        a == b
                        for a, b in zip(signatures[i], signatures[j], strict=True)
                    )
                    / config.minhash_permutations
                )
                jaccard = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
                if (
                    estimate >= config.minhash_threshold
                    and jaccard >= config.minhash_threshold
                ):
                    reason, score = "minhash", jaccard
                else:
                    a, b = (
                        embeddings.vectors[problem_key(left.problem)],
                        embeddings.vectors[problem_key(right.problem)],
                    )
                    norm_a, norm_b = math.hypot(*a), math.hypot(*b)
                    cosine = math.fsum(
                        (x / norm_a) * (y / norm_b) for x, y in zip(a, b, strict=True)
                    )
                    if cosine >= config.embedding_cos:
                        reason, score = "embedding", cosine
            if reason:
                parents[find(i)] = find(j)
                edges.append(
                    {
                        "left": left.prompt_id,
                        "right": right.prompt_id,
                        "method": reason,
                        "score": score,
                    }
                )
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    return tuple(
        tuple(group)
        for group in sorted(groups.values(), key=lambda group: group[0].prompt_id)
    ), edges
