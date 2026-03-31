"""main.py — Entry point for the Weld Inspection Vision System.

Usage:
    python main.py          # spustí GUI
    python main.py --gui    # to isté, explicitný príznak
"""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Weld Inspection Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gui", action="store_true",
                   help="Spustiť grafické rozhranie (predvolené správanie)")
    p.add_argument("--profiles-dir", metavar="DIR", default="config/profiles",
                   help="Adresár s uloženými profilmi (predvolené: config/profiles)")
    return p


def _run_gui(args) -> int:
    from PyQt6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow(profiles_dir=args.profiles_dir)
    win.show()
    win.raise_()
    win.activateWindow()
    return app.exec()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return _run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
