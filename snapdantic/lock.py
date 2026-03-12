"""
lock.py - Single-writer / multi-reader lock for snapdantic.

Two-layer concurrency model:
  Layer 1 (SQLite WAL)  : guarantees consistent reads for multiple concurrent
                          readers without any application-level lock.
  Layer 2 (file lock)   : guarantees that at most one writer process holds the
                          exclusive write lock at any time.

Read processes do NOT acquire any lock; they rely entirely on SQLite WAL
snapshot isolation.

Write processes acquire a file lock at <db_path>.write.lock before any write
operation. The lock is held for the duration of a single write transaction
(encode + flush), then released.

Cross-platform:
  POSIX   → fcntl.flock  (via filelock)
  Windows → msvcrt.locking (via filelock)
"""
from __future__ import annotations

from filelock import FileLock, Timeout


class SnapLock:
    """
    Thin wrapper around filelock.FileLock for snapdantic write serialization.

    Parameters
    ----------
    db_path  : str   - path to the .db file; the lock file is db_path + ".write.lock"
    writable : bool  - if False, acquire() / release() are no-ops (read-only mode)
    timeout  : float - seconds to wait before raising filelock.Timeout (default 30)
    """

    def __init__(self, db_path: str, writable: bool, timeout: float = 30.0) -> None:
        self._writable = writable
        self._lock = (
            FileLock(db_path + ".write.lock", timeout=timeout)
            if writable
            else None
        )

    def acquire(self) -> None:
        """Acquire the write lock. Blocks until the lock is obtained or timeout."""
        if self._writable:
            try:
                self._lock.acquire()
            except Timeout as exc:
                raise TimeoutError(
                    "Could not acquire the write lock within the timeout period. "
                    "Another process may be holding the lock."
                ) from exc

    def release(self) -> None:
        """Release the write lock if currently held."""
        if self._writable and self._lock is not None and self._lock.is_locked:
            self._lock.release()

    def __enter__(self) -> "SnapLock":
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        self.release()
