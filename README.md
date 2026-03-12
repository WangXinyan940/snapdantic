# snapdantic

**Snap**shot + Py**dantic** = Transparent persistent Pydantic models backed by SQLite + Zarr.

Store `numpy` arrays (via Zarr chunked storage), arbitrary Python objects (via pickle), and plain scalars — all in a single `.db` file — by simply subclassing `SnapBase`.

## Quick Start

```python
from snapdantic import SnapBase, PickleField
import pydantic_numpy.typing as pnd
import numpy as np

class TrainingSnapshot(SnapBase):
    epoch:   int               = 0
    loss:    float             = 0.0
    weights: pnd.NpNDArrayFp32 = None   # zarr table
    meta:    PickleField        = None   # blobs table

# Write
ckpt = TrainingSnapshot(db_path="run.db", writable=True)
ckpt.epoch   = 100
ckpt.loss    = 0.001
ckpt.weights = np.random.randn(784, 256).astype(np.float32)
ckpt.meta    = {"config": {"lr": 0.001}}

# Read (concurrent, no lock needed)
ckpt_r = TrainingSnapshot(db_path="run.db", writable=False)
print(ckpt_r.epoch)          # 100
print(ckpt_r.weights.shape)  # (784, 256)

# Clean up orphaned data
removed = ckpt.cleanup()
```

## Installation

```bash
pip install "pydantic>=2.0" "pydantic-numpy>=5.0" "numpy>=1.24" \
            "filelock>=3.12" "zarr>=2.0,<3.0" "numcodecs>=0.11"
pip install -e .
```

> **Note:** zarr v3 has removed `SQLiteStore`. This library is locked to `zarr>=2.0,<3.0`.

## Architecture

- `snapshot_meta` table — Pydantic model JSON (managed by `sqlite3`)
- `blobs` table — pickle objects (managed by `sqlite3`)
- `zarr` table — numpy array chunks (managed by `zarr.SQLiteStore`)
- All three tables share a single `.db` file.

See `architecture.md` in [snapdantic-doc](https://github.com/WangXinyan940/snapdantic-doc) for full design details.

## TypeCodec Plugin

Register custom encode/decode logic for third-party types:

```python
from snapdantic import TypeCodec, CodecRegistry

@CodecRegistry.codec
class MyCodec(TypeCodec):
    @property
    def target_type(self): return MyClass

    def encode(self, obj, db, zarr):
        # store components, return {name: uuid} dict
        ...

    def decode(self, refs, db, zarr):
        # reconstruct from refs
        ...
```

## License

MIT
