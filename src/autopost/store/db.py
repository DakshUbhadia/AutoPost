"""SQLite setup and transactional, checksum-verified forward migrations."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from importlib.resources import files
from pathlib import Path

from autopost.contracts.persistence import METADATA_SCHEMA_VERSION

from .errors import SchemaVersionError

CURRENT_SCHEMA_VERSION = METADATA_SCHEMA_VERSION
MIGRATIONS = {1: "schema.sql", 2: "migrations/002_durable_stores.sql"}


def schema_sql() -> str:
    return files("autopost.store").joinpath("schema.sql").read_text(encoding="utf-8")


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def init_db(path: str | Path = "runs/autopost.sqlite") -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if existing:
            raise RuntimeError(
                f"database already contains tables: {', '.join(row['name'] for row in existing)}"
            )
        _migrate(connection)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def migration_sql(version: int) -> str:
    return (
        files("autopost.store")
        .joinpath(MIGRATIONS[version])
        .read_text(encoding="utf-8")
    )


def _execute_script(connection: sqlite3.Connection, sql: str) -> None:
    # executescript implicitly commits a pending transaction. Execute complete
    # statements ourselves so a failed migration rolls back DDL and its ledger.
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""
    if statement.strip() and not all(
        line.lstrip().startswith("--") or not line.strip()
        for line in statement.splitlines()
    ):
        raise SchemaVersionError("incomplete migration statement")


def _migrate(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version < 0 or version > CURRENT_SCHEMA_VERSION:
        raise SchemaVersionError(f"unsupported metadata schema version: {version}")
    scripts = {number: migration_sql(number) for number in MIGRATIONS}
    checksums = {
        number: hashlib.sha256(sql.encode("utf-8")).hexdigest()
        for number, sql in scripts.items()
    }
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if version == 0:
        if "ap_schema_migrations" in tables:
            raise SchemaVersionError("migration ledger disagrees with user_version")
        expected = set(re.findall(r"CREATE TABLE\s+(\w+)", scripts[1], re.IGNORECASE))
        if tables and not expected <= tables:
            raise SchemaVersionError(
                "unrecognized unversioned database; refusing migration"
            )
        if tables:
            reference = sqlite3.connect(":memory:")
            try:
                reference.executescript(scripts[1])
                for table in sorted(expected):
                    for pragma in ("table_info", "foreign_key_list"):
                        actual = [
                            tuple(row)
                            for row in connection.execute(f"PRAGMA {pragma}({table})")
                        ]
                        known = reference.execute(
                            f"PRAGMA {pragma}({table})"
                        ).fetchall()
                        if actual != known:
                            raise SchemaVersionError(
                                f"unrecognized legacy table definition: {table}"
                            )
            finally:
                reference.close()
        if not tables:
            _execute_script(connection, scripts[1])
        connection.execute(
            "CREATE TABLE ap_schema_migrations (version INTEGER PRIMARY KEY, "
            "sha256 TEXT NOT NULL, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO ap_schema_migrations(version, sha256) VALUES (1, ?)",
            (checksums[1],),
        )
        connection.execute("PRAGMA user_version = 1")
        version = 1
    try:
        applied = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT version, sha256 FROM ap_schema_migrations ORDER BY version"
            )
        }
    except sqlite3.DatabaseError as exc:
        raise SchemaVersionError(
            "metadata migration ledger is missing or corrupt"
        ) from exc
    if applied != {number: checksums[number] for number in range(1, version + 1)}:
        raise SchemaVersionError("metadata migration checksums or versions disagree")
    for target in range(version + 1, CURRENT_SCHEMA_VERSION + 1):
        _execute_script(connection, scripts[target])
        connection.execute(
            "INSERT INTO ap_schema_migrations(version, sha256) VALUES (?, ?)",
            (target, checksums[target]),
        )
        connection.execute(f"PRAGMA user_version = {target}")


def migrate_db(path: str | Path = "runs/autopost.sqlite") -> int:
    connection = connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _migrate(connection)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return CURRENT_SCHEMA_VERSION


def schema_version(path: str | Path) -> int:
    if not Path(path).is_file():
        raise FileNotFoundError("metadata database does not exist")
    connection = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()
