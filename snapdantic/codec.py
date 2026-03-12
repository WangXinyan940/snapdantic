"""
codec.py - TypeCodec plugin mechanism for snapdantic.

TypeCodec is the abstract base class for custom encode/decode logic for
third-party types (e.g. mdtraj.Trajectory). CodecRegistry manages
registration and lookup, following Python's MRO for subclass matching.

The core library defines no concrete codecs — all type-specific logic
lives in external packages (e.g. mdtraj_codecs).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .db import SnapDB
    from .zarr_store import ZarrStore


class TypeCodec(ABC):
    """
    Abstract base class for encoding/decoding third-party Python types.

    Subclasses must implement:
      - target_type : the Python type this codec handles
      - encode()    : decompose obj into UUID references
      - decode()    : reconstruct obj from UUID references

    Optionally override:
      - collect_uuids() : enumerate all UUIDs this codec uses (for cleanup)
    """

    @property
    @abstractmethod
    def target_type(self) -> type:
        """The Python type handled by this codec."""
        ...

    @abstractmethod
    def encode(
        self, obj: Any, db: "SnapDB", zarr: "ZarrStore"
    ) -> dict[str, str | None]:
        """
        Decompose obj into component parts and store each.

        Each ndarray component should be stored via zarr.store_ndarray();
        each Python object component via db.store_pickle().

        Parameters
        ----------
        obj  : the object to encode
        db   : SnapDB   - for storing pickle blobs
        zarr : ZarrStore - for storing numpy arrays

        Returns
        -------
        dict[str, str | None]
            Mapping of {component_name -> UUID | None}.
            None means the component is absent (optional field).
        """
        ...

    @abstractmethod
    def decode(
        self, refs: dict[str, str | None], db: "SnapDB", zarr: "ZarrStore"
    ) -> Any:
        """
        Reconstruct an object from its stored component UUIDs.

        Parameters
        ----------
        refs : dict[str, str | None] - component name → UUID mapping
        db   : SnapDB
        zarr : ZarrStore

        Returns
        -------
        Any : the reconstructed object
        """
        ...

    def collect_uuids(self, refs: dict) -> set[str]:
        """
        Return all UUIDs referenced by this codec's refs dict.

        Used by SnapBase.cleanup() to identify live (non-orphaned) UUIDs.
        The default implementation collects all non-None string values in refs.
        Override this for codecs with nested or non-trivial ref structures.
        """
        return {v for v in refs.values() if isinstance(v, str)}


class CodecRegistry:
    """
    Registry for TypeCodec instances.

    Supports:
    - @CodecRegistry.codec class decorator for register-on-define
    - MRO-based lookup so subclasses inherit parent codecs
    - Key format: "<module>.<qualname>" (fully-qualified class name)
    """

    _registry: dict[str, TypeCodec] = {}

    @classmethod
    def register(cls, codec: TypeCodec) -> None:
        """Register a codec instance for its target_type."""
        key = cls._make_key(codec.target_type)
        cls._registry[key] = codec

    @classmethod
    def get_for_value(cls, obj: Any) -> tuple[str, TypeCodec] | None:
        """
        Find a codec for the given object by searching its MRO.

        Returns
        -------
        (codec_key, codec) tuple, or None if no codec is registered.
        """
        for klass in type(obj).__mro__:
            key = cls._make_key(klass)
            if key in cls._registry:
                return key, cls._registry[key]
        return None

    @classmethod
    def get_by_key(cls, key: str) -> TypeCodec | None:
        """Look up a codec by its fully-qualified key."""
        return cls._registry.get(key)

    @classmethod
    def codec(cls, codec_class: type) -> type:
        """
        Class decorator that registers a codec upon class definition.

        Usage::

            @CodecRegistry.codec
            class MyCodec(TypeCodec):
                ...
        """
        cls.register(codec_class())
        return codec_class

    @staticmethod
    def _make_key(t: type) -> str:
        return f"{t.__module__}.{t.__qualname__}"
