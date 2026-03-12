"""
zarr_store.py - Zarr-backed ndarray storage for snapdantic.

Uses zarr.SQLiteStore to store chunked numpy arrays in the same .db file
as snapshot_meta and blobs, under the table name "zarr".

⚠️  Requires zarr >= 2.0, < 3.0.
    zarr v3 removed SQLiteStore. Do NOT upgrade zarr until an equivalent
    backend is available in v3.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrStore:
    """
    Zarr chunked storage backed by zarr.SQLiteStore.

    Shares the same .db file with SnapDB; zarr operates on the "zarr" table
    while SnapDB uses "snapshot_meta" and "blobs". The tables do not interfere.

    Parameters
    ----------
    db_path    : path to the .db file
    chunk_size : fixed chunk shape for all stored arrays, or None for auto
    compressor : compression codec name: "blosc" (default), "zstd", "lz4",
                 or None / "" for no compression
    """

    def __init__(
        self,
        db_path: str,
        chunk_size: Optional[tuple] = None,
        compressor: str = "blosc",
    ) -> None:
        # zarr.SQLiteStore shares the .db file.
        # In zarr v2.18, the table name is hardcoded to "zarr" internally;
        # the `table` kwarg is NOT accepted — do not pass it.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.store = zarr.SQLiteStore(db_path)
        self.root = zarr.open_group(store=self.store, mode="a")
        self.chunk_size = chunk_size
        self._compressor_name = compressor

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_compressor(self):
        """Return a numcodecs Blosc compressor, or None for no compression."""
        mapping = {
            "blosc": Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
            "zstd":  Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
            "lz4":   Blosc(cname="lz4",  clevel=5, shuffle=Blosc.BITSHUFFLE),
        }
        return mapping.get(self._compressor_name)  # None → no compression

    def _calc_chunk_size(self, shape: tuple, dtype: np.dtype) -> tuple:
        """
        Automatically calculate chunk shape targeting ~1 MB per chunk.

        Strategy: take the N-th root of (target_element_count) for an N-D
        array, then clamp each dimension to [1, dim].
        If the entire array is < 1 MB, use a single chunk (shape itself).
        """
        item_size = np.dtype(dtype).itemsize
        total_bytes = int(np.prod(shape)) * item_size

        # Whole array fits in one chunk
        if total_bytes < 1024 * 1024:
            return shape

        target_elems = 1024 * 1024 // max(item_size, 1)
        ndim = len(shape)
        chunk_per_dim = int(round(target_elems ** (1.0 / ndim)))
        chunk_per_dim = max(1, chunk_per_dim)

        chunks = tuple(min(chunk_per_dim, dim) for dim in shape)
        return chunks

    # ── public API ────────────────────────────────────────────────────────────

    def store_ndarray(self, array: np.ndarray, uuid: str) -> str:
        """
        Store a numpy array with chunked compression under the given UUID.

        The array is stored at root[uuid]; existing data is overwritten.

        Parameters
        ----------
        array : np.ndarray
        uuid  : str - unique key

        Returns
        -------
        str : the uuid (for chaining convenience)
        """
        chunks = self.chunk_size or self._calc_chunk_size(array.shape, array.dtype)
        za = self.root.require_dataset(
            name=uuid,
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks,
            compressor=self._get_compressor(),
            overwrite=True,
        )
        za[...] = array
        return uuid

    def load_ndarray(self, uuid: str) -> np.ndarray:
        """
        Fully load a stored ndarray.

        Raises
        ------
        KeyError : if the UUID is not found
        """
        if uuid not in self.root:
            raise KeyError(f"No ndarray with uuid: {uuid}")
        return self.root[uuid][...]

    def load_ndarray_slice(self, uuid: str, slices: tuple) -> np.ndarray:
        """
        Load only the chunks needed for the given slices (random access).

        Parameters
        ----------
        uuid   : str
        slices : tuple of slice / int / np.ndarray indices

        Raises
        ------
        KeyError : if the UUID is not found
        """
        if uuid not in self.root:
            raise KeyError(f"No ndarray with uuid: {uuid}")
        return self.root[uuid][slices]

    def delete_ndarray(self, uuid: str) -> None:
        """Delete the array and all its chunks for the given UUID."""
        if uuid in self.root:
            del self.root[uuid]

    def list_uuids(self) -> set[str]:
        """Return all UUIDs currently stored in the zarr group."""
        return set(self.root.array_keys())

    def close(self) -> None:
        """Close the underlying SQLiteStore connection."""
        self.store.close()
