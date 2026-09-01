from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    delay_sec: float = 2.0,
    backoff: float = 2.0,
) -> T:
    last_err: Exception | None = None
    wait = delay_sec
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i >= attempts - 1:
                break
            time.sleep(wait)
            wait *= backoff
    assert last_err is not None
    raise last_err
