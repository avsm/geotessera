"""Filesystem helpers shared across the library."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


@contextmanager
def atomic_output(dest: Optional[str | Path], suffix: str = "") -> Iterator[Path]:
    """Yield a temporary path that becomes *dest* on success.

    Any exception, ``KeyboardInterrupt`` included, removes the temporary file
    instead, so a partial write is never observed at *dest* nor left behind as
    a stray ``.*_tmp_*``. The temporary file is created in *dest*'s directory,
    made if needed, to keep the final rename on one filesystem. With
    ``dest=None`` it goes to the system temp directory and is kept on success
    for the caller to own.
    """
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            dir=dest.parent,
            prefix=f".{dest.name}_tmp_",
            suffix=suffix,
            delete=False,
        )
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    path = Path(tmp.name)
    try:
        yield path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    else:
        if dest is not None:
            # replace(), not rename(): rename() raises FileExistsError on
            # Windows when the destination already exists.
            path.replace(dest)
