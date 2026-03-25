"""main.py — CLI entry point for the Weld Inspection Vision System.

Usage examples:

    # Without profile (plain ECC, no ROI, scale = 1 px/mm):
    python main.py --reference data/reference/ref.png --folder data/batch/ --csv results/out.csv

    # With a saved profile:
    python main.py --profile job_01 --folder data/batch/ --json results/out.json --verbose

    # Override reference from a profile:
    python main.py --profile job_01 --reference data/reference/new_ref.png --folder data/batch/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Weld Inspection — batch alignment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--reference",     metavar="PATH",
                   help="Path to reference image (required when --profile is not used)")
    p.add_argument("--folder",        metavar="PATH", required=True,
                   help="Directory containing images to process")
    p.add_argument("--profile",       metavar="NAME",
                   help="Name of a saved profile to load")
    p.add_argument("--profiles-dir",  metavar="DIR", default="config/profiles",
                   help="Directory where profiles are stored (default: config/profiles)")
    p.add_argument("--csv",           metavar="PATH",
                   help="Write results to this CSV file")
    p.add_argument("--json",          metavar="PATH",
                   help="Write results to this JSON file")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-image results to stdout")
    return p


def _run_with_profile(args) -> int:
    from src.config.config_manager import ConfigManager
    from src.batch.batch_processor import process_batch_profile

    cm = ConfigManager(args.profiles_dir)
    try:
        profile = cm.load_profile(args.profile)
    except FileNotFoundError:
        print(f"ERROR: Profile '{args.profile}' not found in '{args.profiles_dir}'",
              file=sys.stderr)
        return 2

    # CLI --reference overrides profile's reference_path
    if args.reference:
        profile.reference_path = args.reference

    if not profile.reference_path:
        print("ERROR: No reference image specified (set in profile or use --reference)",
              file=sys.stderr)
        return 2

    if not Path(profile.reference_path).exists():
        print(f"ERROR: Reference image not found: {profile.reference_path}", file=sys.stderr)
        return 2

    try:
        result = process_batch_profile(
            profile, args.folder,
            export_csv=args.csv,
            export_json=args.json,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    stats = result.stats
    print(f"\nBatch complete — {stats.count_ok}/{stats.count_total} images OK "
          f"({stats.count_error} errors)")
    if stats.count_ok > 0:
        print(f"  dx_px:  mean={stats.dx_px_mean:.3f}  std={stats.dx_px_std:.3f}  "
              f"[{stats.dx_px_min:.3f}, {stats.dx_px_max:.3f}]")
        print(f"  dy_px:  mean={stats.dy_px_mean:.3f}  std={stats.dy_px_std:.3f}  "
              f"[{stats.dy_px_min:.3f}, {stats.dy_px_max:.3f}]")
        print(f"  angle:  mean={stats.angle_deg_mean:.4f}°  std={stats.angle_deg_std:.4f}°")
        print(f"  conf:   mean={stats.confidence_mean:.3f}  "
              f"min={stats.confidence_min:.3f}")

    if args.verbose:
        print()
        for row in result.rows:
            if row["status"] == "OK":
                print(f"  {row['filename']:40s}  "
                      f"dx={row['dx_px']:+.3f}px  dy={row['dy_px']:+.3f}px  "
                      f"angle={row['angle_deg']:+.4f}°  conf={row['confidence']:.3f}")
            else:
                print(f"  {row['filename']:40s}  ERROR")

    if args.csv:
        print(f"\nCSV written to: {args.csv}")
    if args.json:
        print(f"JSON written to: {args.json}")

    return 1 if stats.count_error > 0 else 0


def _run_plain(args) -> int:
    from src.batch.batch_processor import process_batch

    if not args.reference:
        print("ERROR: --reference is required when --profile is not specified", file=sys.stderr)
        return 2

    if not Path(args.reference).exists():
        print(f"ERROR: Reference image not found: {args.reference}", file=sys.stderr)
        return 2

    try:
        results = process_batch(
            args.reference, args.folder,
            export_csv=args.csv,
            export_json=args.json,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    ok     = sum(1 for r in results if r["status"] == "OK")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\nBatch complete — {ok}/{len(results)} images OK ({errors} errors)")

    if args.verbose:
        print()
        for row in results:
            if row["status"] == "OK":
                print(f"  {row['filename']:40s}  "
                      f"dx={row['dx_px']:+.3f}px  dy={row['dy_px']:+.3f}px  "
                      f"angle={row['angle_deg']:+.4f}°  conf={row['confidence']:.3f}")
            else:
                print(f"  {row['filename']:40s}  ERROR")

    if args.csv:
        print(f"\nCSV written to: {args.csv}")
    if args.json:
        print(f"JSON written to: {args.json}")

    return 1 if errors > 0 else 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not Path(args.folder).is_dir():
        print(f"ERROR: Folder not found: {args.folder}", file=sys.stderr)
        return 2

    if args.profile:
        return _run_with_profile(args)
    else:
        return _run_plain(args)


if __name__ == "__main__":
    sys.exit(main())
