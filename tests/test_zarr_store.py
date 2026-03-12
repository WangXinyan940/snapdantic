"""Tests for zarr_store.py (ZarrStore)."""
import numpy as np
import pytest

from snapdantic.zarr_store import ZarrStore


@pytest.fixture
def store(tmp_path):
    return ZarrStore(str(tmp_path / "test.db"))


class TestZarrStore:
    def test_store_and_load_1d(self, store):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        store.store_ndarray(arr, "uuid-1")
        loaded = store.load_ndarray("uuid-1")
        np.testing.assert_array_equal(arr, loaded)

    def test_store_and_load_2d(self, store):
        arr = np.random.randn(100, 50).astype(np.float64)
        store.store_ndarray(arr, "uuid-2d")
        loaded = store.load_ndarray("uuid-2d")
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_store_preserves_dtype(self, store):
        for dtype in [np.float32, np.float64, np.int32, np.int64, np.uint8]:
            arr = np.ones((10,), dtype=dtype)
            store.store_ndarray(arr, f"uuid-dtype-{dtype.__name__}")
            loaded = store.load_ndarray(f"uuid-dtype-{dtype.__name__}")
            assert loaded.dtype == dtype

    def test_load_missing_raises_key_error(self, store):
        with pytest.raises(KeyError):
            store.load_ndarray("nonexistent-uuid")

    def test_overwrite(self, store):
        store.store_ndarray(np.array([1, 2, 3]), "uid")
        store.store_ndarray(np.array([4, 5, 6]), "uid")
        loaded = store.load_ndarray("uid")
        np.testing.assert_array_equal(loaded, [4, 5, 6])

    def test_delete_ndarray(self, store):
        store.store_ndarray(np.zeros(10), "uid-del")
        assert "uid-del" in store.list_uuids()
        store.delete_ndarray("uid-del")
        assert "uid-del" not in store.list_uuids()

    def test_delete_nonexistent_is_noop(self, store):
        store.delete_ndarray("no-such-uuid")  # should not raise

    def test_list_uuids(self, store):
        uids = {"a", "b", "c"}
        for u in uids:
            store.store_ndarray(np.zeros(3), u)
        assert store.list_uuids() == uids

    def test_load_slice(self, store):
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        store.store_ndarray(arr, "uuid-slice")
        sliced = store.load_ndarray_slice("uuid-slice", (slice(0, 3), slice(0, 3)))
        np.testing.assert_array_equal(sliced, arr[:3, :3])

    def test_auto_chunk_small_array(self, store):
        """Arrays < 1 MB should be stored as a single chunk."""
        arr = np.zeros((10, 10), dtype=np.float32)
        chunks = store._calc_chunk_size(arr.shape, arr.dtype)
        assert chunks == arr.shape

    def test_auto_chunk_large_array(self, store):
        """Large arrays should produce multi-chunk shapes."""
        arr_shape = (1000, 1000)
        dtype = np.float32  # 4 bytes/elem → 4 MB total
        chunks = store._calc_chunk_size(arr_shape, np.dtype(dtype))
        # Each chunk should be < total size
        assert chunks != arr_shape
        # Chunk bytes should be close to 1 MB
        item_size = np.dtype(dtype).itemsize
        chunk_bytes = int(np.prod(chunks)) * item_size
        assert chunk_bytes <= 2 * 1024 * 1024  # within 2x of target

    def test_close_does_not_raise(self, store):
        store.close()
