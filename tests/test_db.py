"""Tests for db.py (SnapDB)."""
import os
import tempfile

import pytest

from snapdantic.db import SnapDB


@pytest.fixture
def db(tmp_path):
    return SnapDB(str(tmp_path / "test.db"))


class TestSnapDB:
    # ── pickle / blobs table ─────────────────────────────────────────────────

    def test_store_and_load_pickle(self, db):
        obj = {"key": [1, 2, 3], "nested": {"a": True}}
        uid = "uuid-pickle-1"
        returned = db.store_pickle(obj, uid)
        assert returned == uid
        loaded = db.load_pickle(uid)
        assert loaded == obj

    def test_load_pickle_missing_raises(self, db):
        with pytest.raises(KeyError):
            db.load_pickle("nonexistent-uuid")

    def test_store_pickle_overwrite(self, db):
        uid = "uuid-overwrite"
        db.store_pickle({"v": 1}, uid)
        db.store_pickle({"v": 2}, uid)
        assert db.load_pickle(uid) == {"v": 2}

    def test_get_all_blob_uuids(self, db):
        uids = {"a", "b", "c"}
        for u in uids:
            db.store_pickle(u, u)
        assert db.get_all_blob_uuids() == uids

    def test_delete_blobs(self, db):
        for uid in ["x", "y", "z"]:
            db.store_pickle(uid, uid)
        db.delete_blobs({"x", "z"})
        assert db.get_all_blob_uuids() == {"y"}

    def test_delete_blobs_empty_set(self, db):
        db.store_pickle("obj", "uid1")
        db.delete_blobs(set())  # should be a no-op
        assert db.get_all_blob_uuids() == {"uid1"}

    # ── snapshot_meta table ──────────────────────────────────────────────────

    def test_save_and_load_json(self, db):
        db.save_json("snap1", '{"epoch": 10}')
        assert db.load_json("snap1") == '{"epoch": 10}'

    def test_load_json_missing_returns_none(self, db):
        assert db.load_json("does-not-exist") is None

    def test_save_json_upsert(self, db):
        db.save_json("snap1", '{"epoch": 1}')
        db.save_json("snap1", '{"epoch": 2}')
        assert db.load_json("snap1") == '{"epoch": 2}'

    def test_multiple_snapshots(self, db):
        db.save_json("s1", '{"a": 1}')
        db.save_json("s2", '{"b": 2}')
        assert db.load_json("s1") == '{"a": 1}'
        assert db.load_json("s2") == '{"b": 2}'

    # ── WAL mode ─────────────────────────────────────────────────────────────

    def test_wal_mode_enabled(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "wal.db")
        SnapDB(db_path)  # triggers WAL pragma
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        conn.close()
        assert mode == "wal"
