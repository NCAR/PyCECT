#!/usr/bin/env python3
"""
Check ensemble CESM directories for missing history files.
Optionally delete failed directories or create a history_files directory of symlinks.

Usage:
    python check_for_missing_files.py [path]
    python check_for_missing_files.py [path] --delete
    python check_for_missing_files.py [path] --make-symlinks
    python check_for_missing_files.py [path] --make-symlinks --history-filter cam.h0i
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

CASE_DIR_SUFFIX = re.compile(r"(?<!\d)\d{4}$")


def find_case_dirs(base_path: Path):
    try:
        all_entries = sorted(base_path.iterdir())
    except PermissionError:
        print(f"ERROR: Cannot read directory: {base_path}")
        sys.exit(1)

    return [
        e
        for e in all_entries
        if e.is_dir() and e.name != "history_files" and CASE_DIR_SUFFIX.search(e.name)
    ]


def get_history_files(case_dir: Path, history_filter: str = ""):
    """Return (list_of_nc_paths, error_string). One of the two will be None."""
    run_dir = case_dir / "run"
    if not run_dir.exists():
        return None, "run/ directory does not exist"
    try:
        files = [
            f
            for f in run_dir.iterdir()
            if f.is_file()
            and f.suffix == ".nc"
            and (not history_filter or history_filter in f.name)
        ]
        return files, None
    except PermissionError:
        return None, "Permission denied reading run/"


def check_ensemble_dirs(
    base_path: Path,
    delete: bool = False,
    make_symlinks: bool = False,
    history_filter: str = "cam.h0i",
    force: bool = False,
):
    case_dirs = find_case_dirs(base_path)

    if not case_dirs:
        print(f"No case directories found in: {base_path}")
        sys.exit(1)

    print(f"Found {len(case_dirs)} case directories in {base_path}")
    print(f"History file filter: '{history_filter}'\n")

    missing = []
    present = []
    all_history_files = []

    for case_dir in case_dirs:
        history_files, error = get_history_files(case_dir, history_filter)
        if error:
            missing.append((case_dir.name, error))
        elif history_files:
            present.append((case_dir.name, len(history_files)))
            all_history_files.extend(history_files)
        else:
            missing.append((case_dir.name, f"No .nc files matching '{history_filter}' in run/"))

    print(f"{'=' * 60}")
    print(f"  OK      : {len(present)} directories have matching history files")
    print(f"  MISSING : {len(missing)} directories have no matching history files")
    print(f"{'=' * 60}\n")

    if missing:
        print("Directories missing matching history files:")
        for name, reason in missing:
            print(f"  [MISSING] {name}")
            print(f"            {reason}")
    else:
        print("All case directories have matching history files.")

    if present and missing:
        print(f"\nExample of a good directory: {present[0][0]}")
        print(f"  ({present[0][1]} history file(s) found)")

    if delete and missing:
        print(f"\n{'=' * 60}")
        print(f"  WARNING: About to permanently delete {len(missing)} directories.")
        print(f"{'=' * 60}")
        if not force:
            answer = input("\nType 'yes' to confirm deletion: ").strip().lower()
            if answer != "yes":
                print("Deletion cancelled.")
                return

        deleted = 0
        failed = 0
        print()
        for name, _ in missing:
            case_dir = base_path / name
            try:
                shutil.rmtree(case_dir)
                print(f"  Deleted: {name}")
                deleted += 1
            except Exception as e:
                print(f"  FAILED to delete {name}: {e}")
                failed += 1

        print(f"\nDone. Deleted: {deleted}  Failed: {failed}")

    if make_symlinks:
        print(f"\n{'=' * 60}")
        print(f"  Case directories : {len(case_dirs)}")
        print(f"  Directories with matching history files : {len(present)}")
        print(f"  Total matching history files found : {len(all_history_files)}")
        if missing:
            print(f"  Directories with no matching history files : {len(missing)}")
            print("  Note: some jobs may still be running or have failed.")
        print(f"{'=' * 60}")

        if not all_history_files:
            print("\nNo history files found. Nothing to symlink.")
            return

        if not force:
            answer = (
                input(
                    "\nCreate history_files/ directory with symlinks to all history files? "
                    "Type 'yes' to confirm: "
                )
                .strip()
                .lower()
            )
            if answer != "yes":
                print("Symlink creation cancelled.")
                return

        symlink_dir = base_path / "history_files"
        symlink_dir.mkdir(exist_ok=True)

        created = 0
        skipped = 0
        failed = 0
        for hf in all_history_files:
            link = symlink_dir / hf.name
            if link.exists() or link.is_symlink():
                skipped += 1
                continue
            try:
                link.symlink_to(hf.resolve())
                created += 1
            except Exception as e:
                print(f"  FAILED to create symlink for {hf.name}: {e}")
                failed += 1

        print(f"\nDone. Created: {created}  Skipped (already exist): {skipped}  Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check CESM ensemble directories for missing history files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Base directory containing ensemble cases (default: current directory)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Prompt to delete directories with missing history files",
    )
    parser.add_argument(
        "--make-symlinks",
        action="store_true",
        help="Create a history_files/ directory with symlinks to all found history files",
    )
    parser.add_argument(
        "--history-filter",
        default="cam.h0i",
        metavar="STRING",
        help="Only consider history files whose names contain STRING (default: cam.h0i)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip all confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    base = Path(args.path)
    if not base.exists():
        print(f"ERROR: Path does not exist: {base}")
        sys.exit(1)

    check_ensemble_dirs(
        base,
        delete=args.delete,
        make_symlinks=args.make_symlinks,
        history_filter=args.history_filter,
        force=args.force,
    )
