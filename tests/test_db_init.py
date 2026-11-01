import re
import sqlite3
from pathlib import Path

import pytest

from autopost.store.db import connect, init_db, schema_sql


def test_initializes_schema_and_refuses_existing_database(tmp_path):
    path = tmp_path / "runs/test.sqlite"
    init_db(path)
    names = set(re.findall(r"CREATE TABLE\s+(\w+)", schema_sql(), re.I))
    with connect(path) as db:
        actual = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert names <= actual
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(RuntimeError, match="already contains tables"):
        init_db(path)


def test_task_measurability_constraint(tmp_path):
    path = tmp_path / "test.sqlite"
    init_db(path)
    row = ("t", "c", "s", "v", "skill", "int", 0, 0, "gate_a", None, "problem", "answer")
    with connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO tasks(task_id,cluster_id,source_id,source_ver,skill,answer_type,measurable,low_entropy,split,strat_base_passrate,problem,answer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
        db.execute("INSERT INTO tasks(task_id,cluster_id,source_id,source_ver,skill,answer_type,measurable,low_entropy,split,strat_base_passrate,problem,answer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row[:8] + ("train",) + row[9:])
