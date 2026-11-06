import hashlib
import sqlite3
from contextlib import closing

import pytest

from autopost.store import db as migrations
from autopost.store.db import (
    CURRENT_SCHEMA_VERSION,
    init_db,
    migrate_db,
    schema_sql,
    schema_version,
)
from autopost.store.errors import SchemaVersionError


def legacy_database(path):
    with closing(sqlite3.connect(path)) as db:
        db.executescript(schema_sql())
        db.execute(
            "INSERT INTO prereg(version,sha256) VALUES (?,?)", ("historical", "a" * 64)
        )
        db.execute(
            "INSERT INTO states(state_id,parent_id,path,global_step,trainer_id,config_hash,stack_hash,verifier_version,status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "historical-state",
                None,
                "legacy/checkpoint",
                10,
                "old-trainer",
                "old-config",
                "old-stack",
                "old-verifier",
                "promoted",
            ),
        )
        db.execute(
            "INSERT INTO events(kind,payload) VALUES (?,?)",
            ("historical-event", '{"value": 1}'),
        )
        db.commit()


def snapshot(path):
    with closing(sqlite3.connect(path)) as db:
        return {
            table: db.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("prereg", "states", "events")
        }


def test_legacy_adoption_is_additive_and_never_rewrites_historical_records(tmp_path):
    path = tmp_path / "legacy.sqlite"
    legacy_database(path)
    before = snapshot(path)
    assert schema_version(path) == 0
    assert migrate_db(path) == CURRENT_SCHEMA_VERSION == 2
    assert snapshot(path) == before
    assert migrate_db(path) == 2
    assert snapshot(path) == before
    with closing(sqlite3.connect(path)) as db:
        assert [
            row[0]
            for row in db.execute(
                "SELECT version FROM ap_schema_migrations ORDER BY version"
            )
        ] == [1, 2]


def test_failure_in_forward_migration_rolls_back_ddl_and_version(tmp_path, monkeypatch):
    path = tmp_path / "version1.sqlite"
    legacy_database(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "CREATE TABLE ap_schema_migrations(version INTEGER PRIMARY KEY,sha256 TEXT NOT NULL,applied_at TEXT)"
        )
        db.execute(
            "INSERT INTO ap_schema_migrations VALUES (1,?, 'fixture')",
            (hashlib.sha256(schema_sql().encode()).hexdigest(),),
        )
        db.execute("PRAGMA user_version=1")
        db.commit()
    before = snapshot(path)
    original = migrations.migration_sql
    monkeypatch.setattr(
        migrations,
        "migration_sql",
        lambda version: (
            original(version) + ("\nINVALID MIGRATION;\n" if version == 2 else "")
        ),
    )
    with pytest.raises(sqlite3.DatabaseError):
        migrate_db(path)
    assert schema_version(path) == 1
    assert snapshot(path) == before
    with closing(sqlite3.connect(path)) as db:
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE name='ap_jobs'").fetchone()
            is None
        )
        assert (
            db.execute("SELECT count(*) FROM ap_schema_migrations").fetchone()[0] == 1
        )


def test_unknown_database_and_future_schema_are_rejected_without_new_tables(tmp_path):
    path = tmp_path / "foreign.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE foreign_data(value TEXT)")
        db.commit()
    with pytest.raises(SchemaVersionError, match="unrecognized"):
        migrate_db(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA user_version=99")
        db.commit()
    with pytest.raises(SchemaVersionError, match="unsupported"):
        migrate_db(path)
    with closing(sqlite3.connect(path)) as db:
        assert [
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ] == ["foreign_data"]


def test_migration_ledger_is_checksum_verified(tmp_path):
    path = tmp_path / "metadata.sqlite"
    init_db(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "UPDATE ap_schema_migrations SET sha256=? WHERE version=2", ("f" * 64,)
        )
        db.commit()
    with pytest.raises(SchemaVersionError, match="checksums"):
        migrate_db(path)


def test_fresh_init_records_current_schema_and_keeps_refusal_of_existing_db(tmp_path):
    path = tmp_path / "metadata.sqlite"
    init_db(path)
    assert schema_version(path) == 2
    with pytest.raises(RuntimeError, match="already contains tables"):
        init_db(path)


def test_unrecognized_legacy_columns_are_rejected_without_adopting_history(tmp_path):
    path = tmp_path / "incompatible-legacy.sqlite"
    legacy_database(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("ALTER TABLE states ADD COLUMN unknown_semantics TEXT")
        db.commit()
    before = snapshot(path)
    with pytest.raises(SchemaVersionError, match="legacy table definition"):
        migrate_db(path)
    assert schema_version(path) == 0
    assert snapshot(path) == before
