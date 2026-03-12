"""
snapdantic - Transparent persistent Pydantic models backed by SQLite + Zarr.

Snapshot + Pydantic = persist numpy arrays and arbitrary Python objects
transparently, with single-file storage and multi-reader / single-writer safety.

Quick start::

    from snapdantic import SnapBase, PickleField
    import pydantic_numpy.typing as pnd
    import numpy as np

    class TrainingSnapshot(SnapBase):
        epoch:   int               = 0
        loss:    float             = 0.0
        weights: pnd.NpNDArrayFp32 = None

    ckpt = TrainingSnapshot(db_path="run.db", writable=True)
    ckpt.epoch   = 100
    ckpt.weights = np.random.randn(784, 256).astype(np.float32)
    print(ckpt.epoch)          # 100
    print(ckpt.weights.shape)  # (784, 256)

See the README and architecture.md for full documentation.
"""

from .base import SnapBase
from .codec import CodecRegistry, TypeCodec
from .types import CodecRef, NumpyRef, PickleRef

# PickleField: annotation marker for arbitrary Python objects
# Usage: field: PickleField = None  → stored in blobs table via pickle
from typing import Annotated, Any
PickleField = Annotated[Any, "pickle_blob"]

# NumpyField: plain np.ndarray without pydantic-numpy dtype/shape validation
# For validated numpy fields use pydantic_numpy.typing.NpNDArray* directly
import numpy as np
NumpyField = np.ndarray

__all__ = [
    "SnapBase",
    "TypeCodec",
    "CodecRegistry",
    "NumpyRef",
    "PickleRef",
    "CodecRef",
    "PickleField",
    "NumpyField",
]

__version__ = "0.1.0"
