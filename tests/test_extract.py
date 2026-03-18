#!/usr/bin/env python3
"""
Test runner: parse every file found in the uploads/ directory.

Run from the project root:
    python tests/test_extract.py

Or test a single file:
    python tests/test_extract.py uploads/sample.pdf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_processor import ExtractProcessor


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _hr(char: str = "─", width: int = 70) -> str:
    return char * width


def _preview(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"  … (+{len(text) - max_chars} chars)"


def run_file(file_path: Path) -> bool:
    """Extract a single file and print results.  Returns True on success."""
    print(f"\n{_hr()}")
    print(f"{BOLD}{CYAN}FILE:{RESET} {file_path.name}  ({file_path.stat().st_size:,} bytes)")
    print(_hr())

    t0 = time.perf_counter()
    try:
        docs = ExtractProcessor.extract(str(file_path))
        elapsed = time.perf_counter() - t0

        print(f"{GREEN}✓ OK{RESET}  —  {len(docs)} document(s)  in {elapsed:.3f}s")
        for i, doc in enumerate(docs, 1):
            meta_str = "  ".join(f"{k}={v}" for k, v in doc.metadata.items() if k != "source")
            header = f"  [{i}/{len(docs)}]"
            if meta_str:
                header += f"  {YELLOW}{meta_str}{RESET}"
            print(header)
            print(f"  {_preview(doc.page_content)}")
        return True

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"{RED}✗ FAILED{RESET}  in {elapsed:.3f}s")
        print(f"  {RED}{type(exc).__name__}: {exc}{RESET}")
        return False


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main() -> None:
    if len(sys.argv) > 1:
        # Single-file mode
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # Scan uploads/ directory
        uploads_dir = Path(__file__).parent.parent / "uploads"
        if not uploads_dir.exists():
            print(f"{YELLOW}uploads/ directory not found — creating it.{RESET}")
            uploads_dir.mkdir(parents=True)

        targets = sorted(p for p in uploads_dir.iterdir() if p.is_file())
        if not targets:
            print(f"{YELLOW}No files found in uploads/. Drop some files there and re-run.{RESET}")
            print(f"Supported extensions: {', '.join(ExtractProcessor.supported_extensions())}")
            return

    passed = 0
    failed = 0
    for path in targets:
        ok = run_file(path)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{_hr('═')}")
    print(f"{BOLD}Results:{RESET}  {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}  |  {passed + failed} total")
    print(_hr("═"))


if __name__ == "__main__":
    main()
