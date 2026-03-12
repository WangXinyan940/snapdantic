"""
types.py - Internal reference types for snapdantic.

NumpyRef, PickleRef, CodecRef are Pydantic models used as internal
placeholders in SnapBase fields. They are completely transparent to
the end user — read/write interception in SnapBase resolves them
to/from the actual values automatically.
"""
from pydantic import BaseModel, ConfigDict


class NumpyRef(BaseModel):
    """Reference to a numpy ndarray stored in the zarr table."""
    uuid: str


class PickleRef(BaseModel):
    """Reference to a pickle-serialized object stored in the blobs table."""
    uuid: str


class CodecRef(BaseModel):
    """
    Reference to a complex object stored via a TypeCodec.

    codec_key: fully-qualified class name, e.g. "mdtraj.core.trajectory.Trajectory"
    refs: mapping of {name -> UUID | None} returned by TypeCodec.encode()
    """
    codec_key: str
    refs: dict

    model_config = ConfigDict(arbitrary_types_allowed=True)
