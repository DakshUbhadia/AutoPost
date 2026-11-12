"""Inspectable engineering fixtures, never research data or embeddings."""

from autopost.contracts.datasets import BuildConfig, SourceSpec

from .dedup import EmbeddingTable
from .hidden import HiddenProgram, Parameter
from .normalize import normalize, problem_key
from .sources import SourceRow, SourceSnapshot

QUESTIONS = (
    ("A shop sells 12 notebooks for 4 coins each. What is the total price?", "48"),
    (
        "A rectangular garden is 7 meters wide and 11 meters long. What is its perimeter?",
        "36",
    ),
    ("What is the exact quotient of 21 divided by 8?", "21/8"),
    (
        "A baker fills nine trays with eight buns on each tray. How many buns are packed?",
        "72",
    ),
    (
        "At a museum a visitor first reads a display about ancient coins and then walks past the sculpture hall into the quiet library. In that library there are twenty shelves and every shelf contains fifteen books. Nobody adds or removes any books during the visit. How many books are on those shelves?",
        "300",
    ),
    ("How many minutes are contained in three hours?", "180"),
    (
        "Find the next number after 14 in an arithmetic sequence with common difference 9.",
        "23",
    ),
    ("Evaluate the square of eleven.", "121"),
    ("What is one fourth of eighty?", "20"),
    (
        "A cyclist covers sixty kilometers in two hours. Find the average speed in kilometers per hour.",
        "30",
    ),
    (
        "A triangle has angles forty degrees and sixty degrees. Find its remaining angle in degrees.",
        "80",
    ),
    ("Calculate the greatest common divisor of 84 and 126.", "42"),
    ("Determine the least common multiple of 12 and 15.", "60"),
    (
        "A cube has edge length five centimeters. Compute its volume in cubic centimeters.",
        "125",
    ),
    ("Find the sum of integers from one through ten inclusive.", "55"),
    ("Convert one and a half meters into centimeters.", "150"),
    (
        "An account pays simple interest at five percent on 400 coins for one year. Find the interest.",
        "20",
    ),
    ("Subtract 37 from 100.", "63"),
    ("What is the remainder when 137 is divided by 12?", "5"),
    ("Evaluate two raised to the seventh power.", "128"),
    ("The mean of three numbers is 24. Find their sum.", "72"),
    ("A polygon has six sides. How many diagonals does it have?", "9"),
    ("Reduce the fraction 35 over 49 to its simplest form.", "5/7"),
    ("Add the exact decimal numbers 0.25 and 0.125.", "0.375"),
    ("Solve the equation 3x plus 7 equals 46 for x.", "13"),
    (
        "A die is rolled once. What is the probability of a face greater than four?",
        "1/3",
    ),
    ("How many seconds elapse during eleven minutes?", "660"),
    ("A jar contains 17 red beads and 26 blue beads. Count its beads.", "43"),
    ("Compute the area of a right triangle with perpendicular sides 12 and 9.", "54"),
    ("A clock makes a full revolution. How many degrees does its hand turn?", "360"),
    ("Give the positive square root of two in exact symbolic form.", "sqrt(2)"),
)


def development_inputs(seed: int = 0):
    rows = [
        SourceRow(f"p{index:02}", problem, answer, "fixture-math")
        for index, (problem, answer) in enumerate(QUESTIONS)
    ]
    benchmark_problem = (
        "A brass lock has 12 teeth and gains 3 more. How many teeth does it have?"
    )
    transfer_problem = (
        "A freight vessel loads 14 crates weighing 23 kg each. What is the cargo mass?"
    )
    rows.extend(
        (
            SourceRow("benchmark-copy", benchmark_problem, "15", "count"),
            SourceRow("transfer-copy", transfer_problem, "322", "mass"),
            SourceRow(
                "embedding-copy",
                "Eight buns are placed into each of nine trays. Count all of the buns.",
                "72",
                "fixture-math",
            ),
            SourceRow(
                "minhash-copy",
                QUESTIONS[4][0] + " Please explain.",
                "300",
                "fixture-math",
            ),
            SourceRow(
                "bad-answer",
                "What is a quotient with a zero denominator?",
                "1/0",
                "bad",
            ),
            SourceRow("yes-no", "Is fourteen an even number?", "yes", "bad"),
            SourceRow(
                "choice",
                "Pick an answer: A) three B) four",
                "4",
                "bad",
                multiple_choice=True,
            ),
            SourceRow("blank", " ", "2", "bad"),
        )
    )

    def spec(name, role, family):
        return SourceSpec(
            source_id=name,
            revision="fixture-v1",
            role=role,
            family_id=family,
            license_id="CC0-engineering-fixture",
            fixture_only=True,
        )

    sources = (
        SourceSnapshot(spec("dev-curated", "train", "curated-family"), tuple(rows)),
        SourceSnapshot(
            spec("dev-transfer", "transfer", "transfer-family"),
            (
                SourceRow("t1", transfer_problem, "322", "mass"),
                SourceRow(
                    "t2",
                    "A reservoir initially stores 91 liters and releases 28 liters. What volume remains?",
                    "63",
                    "volume",
                ),
            ),
        ),
        SourceSnapshot(
            spec("derived-transfer", "train", "transfer-family"),
            (
                SourceRow(
                    "derived",
                    "Evaluate the sum of the first twelve positive odd integers.",
                    "144",
                    "sequence",
                ),
            ),
        ),
        SourceSnapshot(
            spec("dev-benchmark", "benchmark", "benchmark-family"),
            (SourceRow("b1", benchmark_problem, "15", "count"),),
        ),
    )
    keys = sorted(
        {
            problem_key(normalize(source, row).problem)
            for source in sources
            for row in source.rows
            if row.source_item_id not in {"bad-answer", "yes-no", "choice", "blank"}
        }
    )
    vectors = {
        key: tuple(float(index == position) for index in range(len(keys)))
        for position, key in enumerate(keys)
    }
    paraphrase = next(row for row in rows if row.source_item_id == "embedding-copy")
    vectors[problem_key(paraphrase.problem)] = vectors[problem_key(QUESTIONS[3][0])]
    # This hand-specified fixture cache tests cosine matching; it is not BGE
    # output and must never be represented as semantic research validation.
    embeddings = EmbeddingTable(
        "fixture-orthogonal-cache", "fixture-v1", vectors, fixture_only=True
    )
    parents = lambda index: (f"dev-curated@fixture-v1:p{index:02}",)
    programs = (
        HiddenProgram(
            program_id="hidden-price",
            template="A shop sells {count} notebooks for {price} coins each. What is the total price?",
            skill="product",
            parent_lineage_keys=parents(0),
            parameters=(
                Parameter(name="count", minimum=11, maximum=50),
                Parameter(name="price", minimum=2, maximum=20),
            ),
            answer_expression="count * price",
            invariants=("count > 0", "price > 0", "answer / price == count"),
        ),
        HiddenProgram(
            program_id="hidden-perimeter",
            template="A rectangular garden is {width} meters wide and {length} meters long. What is its perimeter?",
            skill="geometry",
            parent_lineage_keys=parents(1),
            parameters=(
                Parameter(name="width", minimum=2, maximum=40),
                Parameter(name="length", minimum=2, maximum=60),
            ),
            answer_expression="2 * (width + length)",
            invariants=("width > 0", "length > 0", "answer > width + length"),
        ),
        HiddenProgram(
            program_id="hidden-quotient",
            template="What is the exact quotient of {numerator} divided by {denominator}?",
            skill="rational",
            parent_lineage_keys=parents(2),
            parameters=(
                Parameter(name="numerator", minimum=11, maximum=80),
                Parameter(name="denominator", minimum=2, maximum=19),
            ),
            answer_expression="numerator / denominator",
            invariants=(
                "denominator > 0",
                "numerator > 0",
                "answer * denominator == numerator",
            ),
        ),
    )
    config = BuildConfig(
        seed=seed,
        split_clusters={
            "train": 16,
            "dev": 2,
            "pilot": 3,
            "screen": 2,
            "gate_a": 2,
            "gate_a_reservoir": 2,
            "gate_b": 2,
            "gate_b_reservoir": 2,
            "transfer_guard": 2,
        },
    )
    return sources, embeddings, config, programs
