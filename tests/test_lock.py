"""Tests for lock.py (SnapLock)."""
import pytest

from snapdantic.lock import SnapLock


class TestSnapLock:
    def test_readonly_lock_is_noop(self, tmp_path):
        """Read-only SnapLock should never create a lock file."""
        db_path = str(tmp_path / "test.db")
        lock = SnapLock(db_path, writable=False)
        lock.acquire()
        lock.release()
        import os
        assert not os.path.exists(db_path + ".write.lock")

    def test_writable_lock_creates_file(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        lock = SnapLock(db_path, writable=True)
        lock.acquire()
        import os
        assert os.path.exists(db_path + ".write.lock")
        lock.release()

    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with SnapLock(db_path, writable=True) as lock:
            import os
            assert os.path.exists(db_path + ".write.lock")

    def test_release_without_acquire_is_safe(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        lock = SnapLock(db_path, writable=True)
        lock.release()  # should not raise

    def test_timeout_raises(self, tmp_path):
        """Second writable lock on same file should time out quickly."""
        db_path = str(tmp_path / "test.db")
        lock1 = SnapLock(db_path, writable=True, timeout=0.1)
        lock2 = SnapLock(db_path, writable=True, timeout=0.1)
        lock1.acquire()
        try:
            with pytest.raises(TimeoutError):
                lock2.acquire()
        finally:
            lock1.release()
