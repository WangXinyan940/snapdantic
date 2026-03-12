"""
db.py - SQLite metadata manager for snapdantic.

Manages two tables in the shared .db file:
  - snapshot_meta : stores Pydantic model JSON snapshots
  - blobs         : stores pickle-serialized Python objects

The zarr table is managed separately by zarr.SQLiteStore (zarr_store.py).
Both share the same .db file via independent sqlite3 connections; SQLite
WAL mode ensures concurrent reads are safe.
"""
from __future__ import annotations

import pickle
import sqlite3
import time
from typing import Any


class SnapDB:
    """
    Manages the snapshot_meta and blobs tables in a SQLite .db file.

    One SnapDB instance corresponds to one .db file.
    WAL mode is forced on initialization to enable concurrent multi-reader access.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create tables and enable WAL mode."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshot_meta (
                    snapshot_id  TEXT PRIMARY KEY,
                    json_data    TEXT NOT NULL,
                    updated_at   REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blobs (
                    uuid        TEXT PRIMARY KEY,
                    data        BLOB NOT NULL,
                    created_at  REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meta_updated "
                "ON snapshot_meta(updated_at);"
            )
            conn.commit()

    # ── pickle object storage (blobs table) ──────────────────────────────────

    def store_pickle(self, obj: Any, uuid: str) -> str:
        """
        Serialize obj with pickle.dumps() and store it in the blobs table.

        Parameters
        ----------
        obj  : Any    - the Python object to serialize
        uuid : str    - the UUID key to store it under

        Returns
        -------
        str : the uuid (for chaining convenience)
        """
        data = pickle.dumps(obj)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO blobs (uuid, data, created_at) VALUES (?, ?, ?)",
                (uuid, data, time.time()),
            )
            conn.commit()
        return uuid

    def load_pickle(self, uuid: str) -> Any:
        """
        Load and deserialize a pickle object from the blobs table.

        Raises
        ------
        KeyError : if the uuid is not found
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM blobs WHERE uuid = ?", (uuid,)
            ).fetchone()
        if row is None:
            raise KeyError(f"No blob with uuid: {uuid}")
        return pickle.loads(row[0])

    # ── model JSON storage (snapshot_meta table) ─────────────────────────────

    def save_json(self, snapshot_id: str, json_str: str) -> None:
        """
        Upsert the model JSON snapshot into the snapshot_meta table.

        Parameters
        ----------
        snapshot_id : str - the snapshot identifier
        json_str    : str - JSON-serialized model state
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshot_meta "
                "(snapshot_id, json_data, updated_at) VALUES (?, ?, ?)",
                (snapshot_id, json_str, time.time()),
            )
            conn.commit()

    def load_json(self, snapshot_id: str) -> str | None:
        """
        Load the JSON snapshot for a given snapshot_id.

        Returns
        -------
        str | None : json_data string, or None if not found
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT json_data FROM snapshot_meta WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        return row[0] if row else None

    # ── cleanup support ───────────────────────────────────────────────────────

    def get_all_blob_uuids(self) -> set[str]:
        """Return all UUIDs currently stored in the blobs table."""
        with self._connect() as conn:
            rows = conn.execute("SELECT uuid FROM blobs").fetchall()
        return {row[0] for row in rows}

    def delete_blobs(self, uuids: set[str]) -> None:
        """
        Delete blobs with the given UUIDs.

        Parameters
        ----------
        uuids : set[str] - UUIDs to delete; empty set is a no-op
        """
        if not uuids:
            return
        placeholders = ",".join("?" * len(uuids))
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM blobs WHERE uuid IN ({placeholders})",
                list(uuids),
            )
            conn.commit()
