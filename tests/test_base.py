"""
Tests for base.py (SnapBase) — the core integration layer.
Covers transparent read/write, serialization, cleanup, and permissions.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import pydantic_numpy.typing as pnd

from snapdantic import CodecRegistry, PickleField, SnapBase, TypeCodec
from snapdantic.types import CodecRef, NumpyRef, PickleRef


# ── Simple model fixture ──────────────────────────────────────────────────────

class SimpleSnapshot(SnapBase):
    step:    int               = 0
    loss:    float             = 0.0
    label:   str               = "init"
    weights: pnd.NpNDArrayFp32 = None  # type: ignore[assignment]
    meta:    PickleField        = None


@pytest.fixture
def ckpt(tmp_path):
    return SimpleSnapshot(db_path=str(tmp_path / "test.db"), writable=True)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "shared.db")


# ── Scalar fields ─────────────────────────────────────────────────────────────

class TestScalarFields:
    def test_write_and_read_int(self, ckpt):
        ckpt.step = 42
        assert ckpt.step == 42

    def test_write_and_read_float(self, ckpt):
        ckpt.loss = 0.123
        assert abs(ckpt.loss - 0.123) < 1e-9

    def test_write_and_read_str(self, ckpt):
        ckpt.label = "hello"
        assert ckpt.label == "hello"

    def test_default_values(self, ckpt):
        assert ckpt.step == 0
        assert ckpt.loss == 0.0
        assert ckpt.label == "init"

    def test_readonly_raises(self, db_path):
        ckpt_r = SimpleSnapshot(db_path=db_path, writable=False)
        with pytest.raises(PermissionError):
            ckpt_r.step = 1


# ── Numpy fields ──────────────────────────────────────────────────────────────

class TestNumpyFields:
    def test_write_and_read_array(self, ckpt):
        arr = np.random.randn(10, 10).astype(np.float32)
        ckpt.weights = arr
        loaded = ckpt.weights
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_array_stored_as_numpy_ref(self, ckpt):
        arr = np.ones((5,), dtype=np.float32)
        ckpt.weights = arr
        # Bypass __getattribute__ to inspect the internal Ref
        raw = object.__getattribute__(ckpt, "weights")
        assert isinstance(raw, NumpyRef)

    def test_overwrite_generates_new_uuid(self, ckpt):
        ckpt.weights = np.ones((3,), dtype=np.float32)
        raw1 = object.__getattribute__(ckpt, "weights")
        ckpt.weights = np.zeros((3,), dtype=np.float32)
        raw2 = object.__getattribute__(ckpt, "weights")
        assert raw1.uuid != raw2.uuid

    def test_persistence_across_reload(self, db_path):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # write
        w = SimpleSnapshot(db_path=db_path, writable=True)
        w.weights = arr
        del w
        # re-open read-only
        r = SimpleSnapshot(db_path=db_path, writable=False)
        np.testing.assert_array_equal(r.weights, arr)


# ── Pickle fields ─────────────────────────────────────────────────────────────

class TestPickleFields:
    def test_write_and_read_dict(self, ckpt):
        obj = {"config": {"lr": 0.001, "layers": [64, 64]}}
        ckpt.meta = obj
        assert ckpt.meta == obj

    def test_stored_as_pickle_ref(self, ckpt):
        ckpt.meta = {"x": 1}
        raw = object.__getattribute__(ckpt, "meta")
        assert isinstance(raw, PickleRef)

    def test_arbitrary_object(self, ckpt):
        # Use a picklable object (locally-defined classes can't be pickled)
        import datetime
        obj = datetime.datetime(2026, 1, 1, 12, 0, 0)
        ckpt.meta = obj
        loaded = ckpt.meta
        assert loaded == obj


# ── TypeCodec fields ──────────────────────────────────────────────────────────

class Molecule:
    def __init__(self, atoms: np.ndarray, name: str):
        self.atoms = atoms
        self.name  = name


@pytest.fixture(autouse=False)
def register_molecule_codec():
    """Register a test codec; clean up after test."""
    CodecRegistry._registry.clear()

    @CodecRegistry.codec
    class MoleculeCodec(TypeCodec):
        @property
        def target_type(self):
            return Molecule

        def encode(self, obj: Molecule, db, zarr):
            from uuid import uuid4
            u_atoms = str(uuid4())
            u_name  = str(uuid4())
            zarr.store_ndarray(obj.atoms, u_atoms)
            db.store_pickle(obj.name, u_name)
            return {"atoms_uuid": u_atoms, "name_uuid": u_name}

        def decode(self, refs, db, zarr):
            atoms = zarr.load_ndarray(refs["atoms_uuid"])
            name  = db.load_pickle(refs["name_uuid"])
            return Molecule(atoms, name)

    yield
    CodecRegistry._registry.clear()


class CodecModelSnapshot(SnapBase):
    mol: Any = None


class TestCodecFields:
    def test_codec_encode_decode(self, tmp_path, register_molecule_codec):
        db_path = str(tmp_path / "mol.db")
        mol = Molecule(np.array([1.0, 2.0, 3.0], dtype=np.float64), "water")
        ckpt = CodecModelSnapshot(db_path=db_path, writable=True)
        ckpt.mol = mol

        loaded = ckpt.mol
        assert loaded.name == "water"
        np.testing.assert_array_equal(loaded.atoms, mol.atoms)

    def test_stored_as_codec_ref(self, tmp_path, register_molecule_codec):
        db_path = str(tmp_path / "mol2.db")
        ckpt = CodecModelSnapshot(db_path=db_path, writable=True)
        ckpt.mol = Molecule(np.zeros(5), "co2")
        raw = object.__getattribute__(ckpt, "mol")
        assert isinstance(raw, CodecRef)

    def test_missing_codec_raises(self, tmp_path, register_molecule_codec):
        db_path = str(tmp_path / "mol3.db")
        ckpt = CodecModelSnapshot(db_path=db_path, writable=True)
        ckpt.mol = Molecule(np.zeros(5), "o2")
        # Manually clear registry to simulate missing codec at read time
        CodecRegistry._registry.clear()
        with pytest.raises(RuntimeError, match="No codec registered"):
            _ = ckpt.mol


# ── Persistence / reload ──────────────────────────────────────────────────────

class TestPersistence:
    def test_reload_scalars(self, db_path):
        w = SimpleSnapshot(db_path=db_path, writable=True)
        w.step = 99
        w.label = "test"
        del w
        r = SimpleSnapshot(db_path=db_path)
        assert r.step == 99
        assert r.label == "test"

    def test_snapshot_id_isolation(self, db_path):
        s1 = SimpleSnapshot(db_path=db_path, writable=True, snapshot_id="run_a")
        s1.step = 1
        s2 = SimpleSnapshot(db_path=db_path, writable=True, snapshot_id="run_b")
        s2.step = 2
        del s1, s2

        r1 = SimpleSnapshot(db_path=db_path, snapshot_id="run_a")
        r2 = SimpleSnapshot(db_path=db_path, snapshot_id="run_b")
        assert r1.step == 1
        assert r2.step == 2

    def test_caller_values_override_stored(self, db_path):
        w = SimpleSnapshot(db_path=db_path, writable=True)
        w.step = 5
        del w
        # Re-open with explicit override
        r = SimpleSnapshot(db_path=db_path, step=99)
        # Note: passing step=99 overrides stored value (but r is read-only,
        # so the stored value will be the override on construction)
        assert r.step == 99


# ── Cleanup ───────────────────────────────────────────────────────────────────

class TestCleanup:
    def test_cleanup_removes_orphaned_numpy(self, ckpt):
        ckpt.weights = np.ones((5,), dtype=np.float32)   # uuid-1
        ckpt.weights = np.zeros((5,), dtype=np.float32)  # uuid-2 (uuid-1 orphaned)
        removed = ckpt.cleanup()
        assert removed >= 1

    def test_cleanup_returns_zero_when_clean(self, ckpt):
        ckpt.weights = np.ones((5,), dtype=np.float32)
        # No orphans yet
        removed = ckpt.cleanup()
        assert removed == 0

    def test_cleanup_requires_writable(self, db_path):
        r = SimpleSnapshot(db_path=db_path, writable=False)
        with pytest.raises(PermissionError):
            r.cleanup()

    def test_cleanup_removes_orphaned_pickle(self, ckpt):
        ckpt.meta = {"v": 1}   # uuid-1
        ckpt.meta = {"v": 2}   # uuid-2 (uuid-1 orphaned)
        removed = ckpt.cleanup()
        assert removed >= 1


# ── JSON / YAML export ────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_json_scalars(self, ckpt):
        ckpt.step = 7
        data = json.loads(ckpt.to_json())
        assert data["step"] == 7

    def test_to_json_numpy_ref_structure(self, ckpt):
        ckpt.weights = np.ones((3,), dtype=np.float32)
        data = json.loads(ckpt.to_json())
        assert data["weights"]["__type__"] == "NumpyRef"
        assert "uuid" in data["weights"]

    def test_to_json_pickle_ref_structure(self, ckpt):
        ckpt.meta = {"key": "val"}
        data = json.loads(ckpt.to_json())
        assert data["meta"]["__type__"] == "PickleRef"
        assert "uuid" in data["meta"]

    def test_from_json_round_trip(self, db_path):
        w = SimpleSnapshot(db_path=db_path, writable=True)
        w.step = 42
        w.weights = np.array([1.0, 2.0], dtype=np.float32)
        json_str = w.to_json()
        del w

        r = SimpleSnapshot.from_json(json_str, db_path=db_path)
        assert r.step == 42
        np.testing.assert_array_equal(r.weights, [1.0, 2.0])

    def test_to_yaml_and_from_yaml(self, db_path):
        pytest.importorskip("yaml")
        w = SimpleSnapshot(db_path=db_path, writable=True)
        w.step = 55
        yaml_str = w.to_yaml()
        del w

        r = SimpleSnapshot.from_yaml(yaml_str, db_path=db_path)
        assert r.step == 55

    def test_to_yaml_missing_pyyaml(self, ckpt, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyyaml"):
            ckpt.to_yaml()
