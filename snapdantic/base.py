"""
base.py - SnapBase core: transparent persistent Pydantic model.

SnapBase extends Pydantic's BaseModel with automatic persistence:
  - __setattr__      : encodes the value and flushes to SQLite on every write
  - __getattribute__ : lazily resolves Ref placeholders to their actual values

Storage routing (decided in _encode_value):
  1. TypeCodec match (MRO lookup) → CodecRef  → zarr/blobs (codec decides)
  2. np.ndarray or pydantic-numpy annotation  → NumpyRef  → zarr table
  3. PickleField annotation                   → PickleRef → blobs table
  4. plain scalar                             → stored as-is in JSON

One .db file ↔ one SnapBase instance.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic.fields import PrivateAttr

from .codec import CodecRegistry
from .db import SnapDB
from .lock import SnapLock
from .types import CodecRef, NumpyRef, PickleRef
from .zarr_store import ZarrStore

# pydantic-numpy annotation sentinel — used to detect numpy-typed fields
try:
    from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation as _NpAnnotation
    _HAS_PYDANTIC_NUMPY = True
except ImportError:
    _NpAnnotation = None
    _HAS_PYDANTIC_NUMPY = False


class SnapBase(BaseModel):
    """
    Transparent persistent Pydantic model.

    Usage::

        class TrainingSnapshot(SnapBase):
            epoch:   int               = 0
            loss:    float             = 0.0
            weights: pnd.NpNDArrayFp32 = None   # stored in zarr table
            meta:    PickleField        = None   # stored in blobs table

        ckpt = TrainingSnapshot(db_path="run.db", writable=True)
        ckpt.epoch   = 100
        ckpt.weights = np.random.randn(784, 256).astype(np.float32)

    Parameters
    ----------
    db_path     : path to the .db file (created if it doesn't exist)
    writable    : if False (default), writes raise PermissionError
    snapshot_id : identifier for this snapshot; allows multiple snapshots
                  per .db file (default "default")
    **data      : initial field values (override any stored snapshot)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── private runtime state (excluded from Pydantic serialization) ──────────
    _db:          SnapDB    = PrivateAttr()
    _zarr:        ZarrStore = PrivateAttr()
    _lock:        SnapLock  = PrivateAttr()
    _writable:    bool      = PrivateAttr(default=False)
    _snapshot_id: str       = PrivateAttr(default="default")

    # ── initialization ────────────────────────────────────────────────────────

    def __init__(
        self, *,
        db_path:     str,
        writable:    bool = False,
        snapshot_id: str  = "default",
        **data,
    ) -> None:
        db   = SnapDB(db_path)
        zarr = ZarrStore(db_path)

        # Load existing snapshot from SQLite, then let caller-supplied values
        # override individual fields.
        existing_json = db.load_json(snapshot_id)
        if existing_json:
            stored = _reconstruct_refs(json.loads(existing_json))
            stored.update(data)
            data = stored

        # Use model_construct to bypass ALL pydantic validators.
        # This is necessary because:
        #   - NumpyRef / PickleRef / CodecRef are not valid values for
        #     pydantic-numpy annotated fields (e.g. NpNDArrayFp32)
        #   - None is not valid for non-optional numpy fields at init time
        # We restore full transparency via __setattr__ and __getattribute__.
        #
        # model_construct() sets __dict__ and __pydantic_fields_set__ but
        # does NOT call validators. We then copy those values to self.

        # Build full data dict: merge defaults with caller data
        field_defaults = {
            fname: field.default
            for fname, field in type(self).model_fields.items()
            if field.default is not None or not field.is_required()
        }
        merged = {**field_defaults, **data}

        constructed = type(self).model_construct(**merged)

        # Bootstrap the Pydantic instance without running validators
        # by calling __init__ with no data, then overwriting fields.
        # We temporarily set a sentinel so _flush won't run during this init.
        object.__setattr__(self, "_writable", False)  # pre-init sentinel
        BaseModel.__init__(self)  # initialises __pydantic_private__, etc.

        # Overwrite each field directly, bypassing __setattr__ and validators
        for key in type(self).model_fields:
            val = constructed.__dict__.get(key)
            object.__setattr__(self, key, val)

        # Use object.__setattr__ to bypass our custom __setattr__ for privates
        object.__setattr__(self, "_db",          db)
        object.__setattr__(self, "_zarr",        zarr)
        object.__setattr__(self, "_lock",        SnapLock(db_path, writable))
        object.__setattr__(self, "_writable",    writable)
        object.__setattr__(self, "_snapshot_id", snapshot_id)

    # ── transparent write interception ───────────────────────────────────────

    def __setattr__(self, name: str, value: Any) -> None:
        # Private attributes bypass encoding entirely
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if not object.__getattribute__(self, "_writable"):
            sid = object.__getattribute__(self, "_snapshot_id")
            raise PermissionError(
                f"Snapshot '{sid}' is read-only. Pass writable=True to enable writes."
            )

        # Acquire write lock; encode value and flush all inside the same lock
        lock = object.__getattribute__(self, "_lock")
        with lock:
            encoded = self._encode_value(name, value)
            super().__setattr__(name, encoded)
            self._flush()

    def _encode_value(self, name: str, value: Any) -> Any:
        """
        Convert a user-supplied value to its internal Ref form.
        Called exclusively from within the write lock.
        """
        db   = object.__getattribute__(self, "_db")
        zarr = object.__getattribute__(self, "_zarr")

        # ① TypeCodec — highest priority (complex third-party types)
        codec_result = CodecRegistry.get_for_value(value)
        if codec_result is not None:
            codec_key, codec = codec_result
            refs = codec.encode(value, db, zarr)
            return CodecRef(codec_key=codec_key, refs=refs)

        # ② numpy ndarray (runtime isinstance + pydantic-numpy annotation)
        if isinstance(value, np.ndarray) or self._get_field_kind(name) == "numpy":
            uid = str(uuid4())
            zarr.store_ndarray(value, uid)
            return NumpyRef(uuid=uid)

        # ③ PickleField annotation
        if self._get_field_kind(name) == "pickle":
            uid = str(uuid4())
            db.store_pickle(value, uid)
            return PickleRef(uuid=uid)

        # ④ Plain scalar / None — stored directly in JSON
        return value

    # ── transparent read interception ────────────────────────────────────────

    def __getattribute__(self, name: str) -> Any:
        # Only intercept fields declared in model_fields; leave everything
        # else (methods, private attrs, Pydantic internals) untouched.
        model_fields = super().__getattribute__("model_fields")
        if name not in model_fields:
            return super().__getattribute__(name)

        value = super().__getattribute__(name)
        db    = object.__getattribute__(self, "_db")
        zarr  = object.__getattribute__(self, "_zarr")

        if isinstance(value, NumpyRef):
            return zarr.load_ndarray(value.uuid)

        if isinstance(value, PickleRef):
            return db.load_pickle(value.uuid)

        if isinstance(value, CodecRef):
            codec = CodecRegistry.get_by_key(value.codec_key)
            if codec is None:
                raise RuntimeError(
                    f"No codec registered for '{value.codec_key}'. "
                    "Did you forget to import the codec module?"
                )
            return codec.decode(value.refs, db, zarr)

        return value

    # ── field kind detection ──────────────────────────────────────────────────

    def _get_field_kind(self, name: str) -> str:
        """
        Return 'numpy' | 'pickle' | 'plain' based on field annotation metadata.

        Checks (in order):
          1. pydantic-numpy NpArrayPydanticAnnotation in field.metadata
          2. "pickle_blob" sentinel string in field.metadata
          3. field.annotation is np.ndarray
        """
        field = self.model_fields.get(name)
        if field is None:
            return "plain"

        for meta in (field.metadata or []):
            if _HAS_PYDANTIC_NUMPY and isinstance(meta, _NpAnnotation):
                return "numpy"
            if meta == "pickle_blob":
                return "pickle"

        if field.annotation is np.ndarray:
            return "numpy"

        return "plain"

    # ── serialization (single source of truth) ───────────────────────────────

    def _serialize_to_dict(self) -> dict:
        """
        Serialize the current model state to a plain dict.

        Ref types are represented as dicts with a "__type__" key so they can
        be round-tripped through JSON without losing type information.

        Used by _flush(), to_json(), and to_yaml() — the only place
        serialization logic lives.
        """
        raw = {}
        for fname in self.model_fields:
            # Bypass __getattribute__ to get the internal Ref, not the resolved value
            val = super().__getattribute__(fname)
            if isinstance(val, (NumpyRef, PickleRef, CodecRef)):
                d = val.model_dump()
                d["__type__"] = type(val).__name__
                raw[fname] = d
            else:
                raw[fname] = val
        return raw

    # ── persistence flush ─────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Write the current model JSON to the snapshot_meta table (inside write lock)."""
        db  = object.__getattribute__(self, "_db")
        sid = object.__getattribute__(self, "_snapshot_id")
        db.save_json(sid, json.dumps(self._serialize_to_dict()))

    # ── cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """
        Delete orphaned zarr arrays and pickle blobs not referenced by any field.

        Must be called with writable=True. Runs entirely within the write lock
        to prevent TOCTOU race conditions.

        Returns
        -------
        int : total number of orphaned entries deleted (zarr + blobs combined)

        Notes
        -----
        One .db file corresponds to one SnapBase instance.
        cleanup() considers all data in the file; any UUID not referenced by
        the current instance's fields is considered orphaned.
        """
        if not object.__getattribute__(self, "_writable"):
            raise PermissionError("cleanup() requires writable=True")

        db   = object.__getattribute__(self, "_db")
        zarr = object.__getattribute__(self, "_zarr")
        lock = object.__getattribute__(self, "_lock")

        with lock:
            referenced_zarr:  set[str] = set()
            referenced_blobs: set[str] = set()

            for fname in self.model_fields:
                val = super().__getattribute__(fname)

                if isinstance(val, NumpyRef):
                    referenced_zarr.add(val.uuid)

                elif isinstance(val, PickleRef):
                    referenced_blobs.add(val.uuid)

                elif isinstance(val, CodecRef):
                    codec = CodecRegistry.get_by_key(val.codec_key)
                    uuids = (
                        codec.collect_uuids(val.refs)
                        if codec is not None
                        else {v for v in val.refs.values() if isinstance(v, str)}
                    )
                    # collect_uuids() returns all UUIDs regardless of whether
                    # they live in zarr or blobs; add them to both sets so
                    # neither table accidentally orphans them.
                    referenced_zarr  |= uuids
                    referenced_blobs |= uuids

            # --- zarr orphans ---
            all_zarr  = zarr.list_uuids()
            orphan_zarr = all_zarr - referenced_zarr
            for uid in orphan_zarr:
                zarr.delete_ndarray(uid)

            # --- blob orphans ---
            all_blobs = db.get_all_blob_uuids()
            orphan_blobs = all_blobs - referenced_blobs
            db.delete_blobs(orphan_blobs)

        return len(orphan_zarr) + len(orphan_blobs)

    # ── JSON / YAML export / import ───────────────────────────────────────────

    def to_json(self) -> str:
        """
        Export the model as JSON with UUID references (no raw binary data).

        Safe to transmit, store in version control, or use for inspection.
        Requires the same .db file to restore actual array/blob data.
        """
        return json.dumps(self._serialize_to_dict(), indent=2)

    def to_yaml(self) -> str:
        """
        Export the model as YAML with UUID references.

        Requires pyyaml: pip install snapdantic[yaml]
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML support. "
                "Install it with: pip install 'snapdantic[yaml]'"
            ) from exc
        return yaml.dump(self._serialize_to_dict(), allow_unicode=True)

    @classmethod
    def from_json(
        cls,
        json_str: str, *,
        db_path:     str,
        writable:    bool = False,
        snapshot_id: str  = "default",
    ):
        """
        Restore a SnapBase instance from an exported JSON string.

        Parameters
        ----------
        json_str    : str  - output of to_json()
        db_path     : str  - path to the .db file holding the actual data
        writable    : bool - open in write mode
        snapshot_id : str  - snapshot identifier
        """
        data = _reconstruct_refs(json.loads(json_str))
        return cls(
            db_path=db_path,
            writable=writable,
            snapshot_id=snapshot_id,
            **data,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_str: str, *,
        db_path:     str,
        writable:    bool = False,
        snapshot_id: str  = "default",
    ):
        """
        Restore a SnapBase instance from an exported YAML string.

        Parameters
        ----------
        yaml_str    : str  - output of to_yaml()
        db_path     : str  - path to the .db file holding the actual data
        writable    : bool - open in write mode
        snapshot_id : str  - snapshot identifier
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML support. "
                "Install it with: pip install 'snapdantic[yaml]'"
            ) from exc
        data = _reconstruct_refs(yaml.safe_load(yaml_str))
        return cls(
            db_path=db_path,
            writable=writable,
            snapshot_id=snapshot_id,
            **data,
        )


# ── module-level helper ───────────────────────────────────────────────────────

def _reconstruct_refs(data: dict) -> dict:
    """
    Convert a _serialize_to_dict() output back into NumpyRef / PickleRef /
    CodecRef objects so that SnapBase.__init__ can accept them directly.
    """
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            t = v.get("__type__")
            if t == "NumpyRef":
                result[k] = NumpyRef(uuid=v["uuid"])
            elif t == "PickleRef":
                result[k] = PickleRef(uuid=v["uuid"])
            elif t == "CodecRef":
                result[k] = CodecRef(codec_key=v["codec_key"], refs=v["refs"])
            else:
                result[k] = v
        else:
            result[k] = v
    return result
