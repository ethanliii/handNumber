"""Read individual members out of a huge remote .zip with HTTP range requests.

HaGRID ships images as ~40 GB zips per gesture. To evaluate on real photos we only
need a handful of them, so we read the zip's central directory and then fetch just
the bytes of the files we want.
"""

from __future__ import annotations

import io
import time
import urllib.request


def _request(req: urllib.request.Request, timeout: float) -> tuple[dict, bytes]:
    """(headers, body) with retries: the HaGRID server occasionally resets connections."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return dict(r.headers), r.read()
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


class RangeFile(io.RawIOBase):
    """A seekable, read-only file object backed by HTTP range requests."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        self.block = block
        req = urllib.request.Request(url, method="HEAD")
        headers, _ = _request(req, timeout=60)
        self.size = int(headers["Content-Length"])
        self.pos = 0
        self._cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        return _request(req, timeout=120)[1]

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.block
            self._cache[idx] = self._fetch(start, min(start + self.block, self.size))
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # Large reads go straight to the network instead of through the block cache.
        if n > 4 * self.block:
            data = self._fetch(self.pos, self.pos + n)
            self.pos += n
            return data
        out = bytearray()
        while len(out) < n:
            idx, off = divmod(self.pos, self.block)
            chunk = self._block(idx)[off : off + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
