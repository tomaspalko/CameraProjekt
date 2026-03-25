#!/usr/bin/env python
"""
watch_tests.py — Run tests on every source change. Retry until all pass.

Usage:
    python watch_tests.py              # watch src/ and tests/
    python watch_tests.py --fast       # unit tests only (skip integration)
    python watch_tests.py --once       # run once and exit
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog...")
    subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"], check=True)
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler


# ── Colours ──────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests(fast: bool = False) -> bool:
    """Run pytest. Returns True if all tests passed."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=short",          # concise tracebacks
        "-q",                  # quiet output
        "--no-header",
        "-x",                  # stop on first failure (fast feedback)
        "--color=yes",
    ]

    if fast:
        cmd += ["tests/unit/"]
    else:
        cmd += ["tests/unit/", "tests/integration/"]

    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"{CYAN}▶  Running {'unit' if fast else 'all'} tests  "
          f"[{datetime.now().strftime('%H:%M:%S')}]{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")

    result = subprocess.run(cmd)
    passed = result.returncode == 0

    if passed:
        print(f"\n{GREEN}{BOLD}✅  ALL TESTS PASSED{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}❌  TESTS FAILED — fix and save to retry{RESET}\n")

    return passed


def run_until_pass(fast: bool = False, max_retries: int = 0):
    """
    Run tests in a loop, waiting for file changes between attempts.
    If max_retries == 0, loop forever until tests pass.
    """
    attempt = 0
    while True:
        attempt += 1
        if max_retries and attempt > max_retries:
            print(f"{RED}Max retries ({max_retries}) reached. Exiting.{RESET}")
            sys.exit(1)

        passed = run_tests(fast=fast)
        if passed:
            return

        print(f"{YELLOW}Waiting for file change to retry...{RESET}")
        wait_for_change(watch_dirs=["src", "tests"])


# ── File watcher ──────────────────────────────────────────────────────────────

_change_detected = False


class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global _change_detected
        if event.src_path.endswith(".py"):
            _change_detected = True

    def on_created(self, event):
        global _change_detected
        if event.src_path.endswith(".py"):
            _change_detected = True


def wait_for_change(watch_dirs: list[str]):
    """Block until a .py file changes in the watched directories."""
    global _change_detected
    _change_detected = False

    observer = Observer()
    handler  = ChangeHandler()

    for d in watch_dirs:
        path = Path(d)
        if path.exists():
            observer.schedule(handler, str(path), recursive=True)

    observer.start()
    try:
        while not _change_detected:
            time.sleep(0.3)
    finally:
        observer.stop()
        observer.join()


# ── Watch mode (continuous) ───────────────────────────────────────────────────

def watch_mode(fast: bool = False):
    """Run tests on every file change, forever."""
    print(f"{CYAN}{BOLD}👁  Watch mode active — watching src/ and tests/{RESET}")
    print(f"{CYAN}    Press Ctrl+C to stop{RESET}\n")

    run_tests(fast=fast)  # run once on startup

    while True:
        wait_for_change(watch_dirs=["src", "tests"])
        run_tests(fast=fast)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test watcher for weld_vision")
    parser.add_argument("--fast",        action="store_true",
                        help="Run unit tests only")
    parser.add_argument("--once",        action="store_true",
                        help="Run once and exit (non-zero on failure)")
    parser.add_argument("--until-pass",  action="store_true",
                        help="Retry on file change until tests pass, then exit")
    parser.add_argument("--max-retries", type=int, default=0,
                        help="Max retries in --until-pass mode (0 = infinite)")
    args = parser.parse_args()

    if args.once:
        ok = run_tests(fast=args.fast)
        sys.exit(0 if ok else 1)

    elif args.until_pass:
        run_until_pass(fast=args.fast, max_retries=args.max_retries)
        print(f"{GREEN}Tests passed. Exiting.{RESET}")

    else:
        try:
            watch_mode(fast=args.fast)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Watch mode stopped.{RESET}")


if __name__ == "__main__":
    main()
