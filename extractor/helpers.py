import concurrent.futures
from typing import NamedTuple

import charset_normalizer


class FileEncoding(NamedTuple):
    encoding: str | None
    confidence: float
    language: str | None


def detect_file_encodings(file_path: str, timeout: int = 5) -> list[FileEncoding]:
    """Detect file encoding, returns list ordered by confidence."""

    def _detect(path: str) -> list[FileEncoding]:
        rst = charset_normalizer.from_path(path)
        best = rst.best()
        if best is None:
            return []
        return [FileEncoding(encoding=best.encoding, confidence=best.coherence, language=best.language)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_detect, file_path)
        try:
            encodings = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout while detecting encoding for {file_path}")

    if not encodings or all(e.encoding is None for e in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")
    return [e for e in encodings if e.encoding is not None]
