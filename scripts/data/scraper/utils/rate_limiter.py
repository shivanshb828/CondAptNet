"""
Per-source configurable rate limiter.

Each source in RATE_LIMITS gets its own token bucket enforced via
threading.Lock + time.monotonic(). Thread-safe: safe for concurrent adapters.

Usage:
    from scripts.data.scraper.utils.rate_limiter import get_limiter

    with get_limiter("pubmed"):
        response = requests.get(url)

    # Or explicit call:
    get_limiter("openalex").wait()
    response = requests.get(url)
"""

from __future__ import annotations

import time
import threading
from typing import Dict


class RateLimiter:
    """
    Simple token-bucket rate limiter. Enforces minimum interval between calls.
    Thread-safe via an internal lock.
    """

    def __init__(self, requests_per_second: float, source: str = "") -> None:
        if requests_per_second <= 0:
            raise ValueError(f"requests_per_second must be > 0, got {requests_per_second}")
        self._min_interval = 1.0 / requests_per_second
        self._last_call    = 0.0
        self._lock         = threading.Lock()
        self.source        = source
        self.total_calls   = 0
        self.total_waited  = 0.0

    def wait(self) -> float:
        """
        Block until the minimum interval since the last call has elapsed.
        Returns the seconds waited (0.0 if no wait was needed).
        """
        with self._lock:
            now       = time.monotonic()
            remaining = self._min_interval - (now - self._last_call)
            waited    = 0.0
            if remaining > 0:
                time.sleep(remaining)
                waited = remaining
            self._last_call   = time.monotonic()
            self.total_calls += 1
            self.total_waited += waited
        return waited

    def __enter__(self) -> "RateLimiter":
        self.wait()
        return self

    def __exit__(self, *_args) -> None:
        pass

    def __repr__(self) -> str:
        return (f"RateLimiter(source={self.source!r}, "
                f"interval={self._min_interval:.3f}s, "
                f"calls={self.total_calls})")


# ── Global registry — one limiter per source name ────────────────────────────
_registry: Dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def get_limiter(source: str) -> RateLimiter:
    """
    Return the shared RateLimiter for the given source name.
    Creates one on first call, using RATE_LIMITS from scraper/config.py.
    """
    with _registry_lock:
        if source not in _registry:
            from scripts.data.scraper.config import RATE_LIMITS
            rps = RATE_LIMITS.get(source, 1.0)
            _registry[source] = RateLimiter(rps, source=source)
        return _registry[source]


def reset_all() -> None:
    """Reset the global registry. Used in tests to get fresh limiters."""
    with _registry_lock:
        _registry.clear()


if __name__ == "__main__":
    import sys

    print("RateLimiter self-test (2 req/s limiter, 5 calls):")
    lim = RateLimiter(2.0, source="test")
    times = []
    for i in range(5):
        t0 = time.monotonic()
        lim.wait()
        times.append(time.monotonic() - t0)
        print(f"  call {i+1}: waited {times[-1]:.3f}s")

    # First call should be instant, subsequent calls ~0.5s apart
    assert times[0] < 0.1,  f"First call should be instant, got {times[0]:.3f}s"
    for t in times[1:]:
        assert 0.4 <= t <= 0.7, f"Expected ~0.5s gap, got {t:.3f}s"

    # Context manager test
    start = time.monotonic()
    with RateLimiter(10.0):
        pass
    with RateLimiter(10.0):
        pass
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Two 10 req/s calls should take <1s, got {elapsed:.3f}s"

    print("All RateLimiter tests passed.")
    sys.exit(0)
