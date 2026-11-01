"""SQLite storage setup."""
from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path


def schema_sql() -> str:
    return files("autopost.store").joinpath("schema.sql").read_text(encoding="utf-8")


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_db(path: str | Path = "runs/autopost.sqlite") -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path)
    try:
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if existing:
            raise RuntimeError(f"database already contains tables: {', '.join(row['name'] for row in existing)}")
        connection.executescript(schema_sql())
        connection.commit()
    finally:
        connection.close()
