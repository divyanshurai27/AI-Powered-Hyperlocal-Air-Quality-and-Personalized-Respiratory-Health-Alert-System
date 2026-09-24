"""Shared HTTP behaviour for source adapters: timeouts, bounded retries, typed failures."""

import time
from collections.abc import Callable
from typing import Any

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

# 408: OpenAQ times out heavy queries server-side; worth retrying (and chunking).
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_RETRY_WAIT_SECONDS = 120.0


class SourceError(Exception):
    """The source responded, but not with something we can use (bad key, malformed body...)."""

    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source


class SourceUnavailable(SourceError):
    """Timeout, connection failure, or retryable errors that persisted past the retry budget."""


def request_json(
    client: httpx.Client,
    source: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET `url` and return parsed JSON, retrying transient failures with exponential backoff."""
    problem = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        delay = backoff_seconds * 2 ** (attempt - 1)
        try:
            response = client.get(url, params=params)
        except httpx.TimeoutException:
            problem = "timed out"
        except httpx.TransportError as exc:
            problem = f"connection error: {type(exc).__name__}"
        else:
            if response.status_code in RETRYABLE_STATUS:
                problem = f"HTTP {response.status_code}"
                delay = _retry_after(response) or delay
            elif response.status_code in (401, 403):
                raise SourceError(source, f"HTTP {response.status_code}: check the API key")
            elif response.status_code >= 400:
                raise SourceError(source, f"HTTP {response.status_code}: {response.text[:200]}")
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise SourceError(source, "response body is not valid JSON") from exc

        if attempt < max_attempts:
            logger.warning(
                "source_retry",
                extra={"source": source, "attempt": attempt, "problem": problem, "delay_s": delay},
            )
            sleep(delay)
    raise SourceUnavailable(source, f"{problem} after {max_attempts} attempts")


def _retry_after(response: httpx.Response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset"):
        value = response.headers.get(header)
        if value:
            try:
                return min(max(float(value), 0.0), MAX_RETRY_WAIT_SECONDS)
            except ValueError:
                return None
    return None
