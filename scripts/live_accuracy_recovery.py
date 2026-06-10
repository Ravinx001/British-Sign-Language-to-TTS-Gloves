"""Guardrails for the live hardware accuracy recovery workflow.

This script intentionally checks only the Ravindu real-data folder used by
scripts.run_full_user_validation. It prevents a retrain from running before the
extra live correction sessions from the recovery plan have been collected.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_USER_DIR = PROJECT_ROOT / "data" / "raw" / "real" / "ravindu-8dd5c02e"
USER_ID = "8dd5c02e"

CORRECTION_TARGETS = {
    "U": 60,
    "O": 60,
    "N": 60,
    "H": 60,
    "J": 60,
    "I": 45,
    "V": 45,
    "M": 45,
    "Y": 45,
    "L": 45,
}

PROBLEM_VALIDATION_COMMAND = (
    ".\\.venv\\Scripts\\python.exe -m scripts.validate_live_recognition "
    "--new-user --letters I,U,O,N,L,H,J --reps 5 --hold 3 --live-debug "
    "--target-top1 0.80 --target-top3 0.91 --min-correct-per-letter 4 "
    "--out data/figures/live_validation_problem_letters_after_retrain.json"
)

ALL_LETTERS_VALIDATION_COMMAND = (
    ".\\.venv\\Scripts\\python.exe -m scripts.validate_live_recognition "
    "--new-user --letters A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z "
    "--reps 3 --hold 3 --live-debug "
    "--out data/figures/live_validation_all_letters_after_retrain.json"
)


def count_sessions(label: str) -> int:
    label_dir = RAW_USER_DIR / label
    if not label_dir.exists():
        return 0
    return sum(1 for path in label_dir.glob("*.jsonl") if path.is_file())


def recovery_counts() -> dict[str, dict[str, int]]:
    rows = {}
    for label, target in CORRECTION_TARGETS.items():
        current = count_sessions(label)
        rows[label] = {
            "current": current,
            "target": target,
            "missing": max(0, target - current),
        }
    return rows


def ready_for_retrain(rows: dict[str, dict[str, int]]) -> bool:
    return all(row["missing"] == 0 for row in rows.values())


def print_status(rows: dict[str, dict[str, int]]) -> None:
    print("=================================================================")
    print("Live hardware accuracy recovery status")
    print("=================================================================")
    print(f"Raw user folder: {RAW_USER_DIR}")
    print(f"User id: {USER_ID}")
    print()
    print("Correction-session targets:")
    print("  Letter  Current  Target  Missing")
    for label in CORRECTION_TARGETS:
        row = rows[label]
        print(
            f"  {label:<6}  {row['current']:>7}  "
            f"{row['target']:>6}  {row['missing']:>7}"
        )
    print()
    if ready_for_retrain(rows):
        print("Status: READY for Ravindu-only retraining.")
        print()
        print("Run:")
        print("  .\\.venv\\Scripts\\python.exe -m scripts.live_accuracy_recovery train")
    else:
        print("Status: NOT READY for retraining.")
        print("Collect the missing sessions at http://localhost:8000/collection")
        print("using user ravindu / 8dd5c02e, then rerun this status command.")
    print()
    print("After a successful retrain, validate problem letters with:")
    print(f"  {PROBLEM_VALIDATION_COMMAND}")
    print()
    print("Then validate all letters with:")
    print(f"  {ALL_LETTERS_VALIDATION_COMMAND}")


def run_train() -> int:
    rows = recovery_counts()
    print_status(rows)
    if not ready_for_retrain(rows):
        print()
        print("Blocked: correction data is incomplete, so retraining was not run.")
        return 2

    print()
    print("Starting Ravindu-only full validation and deployment run...")
    cmd = [sys.executable, "-m", "scripts.run_full_user_validation"]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check and run the Ravindu-only live accuracy recovery workflow."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Show missing correction sessions")
    subparsers.add_parser(
        "train",
        help="Run Ravindu-only retraining only when correction data is complete",
    )
    args = parser.parse_args()

    command = args.command or "status"
    if command == "status":
        print_status(recovery_counts())
        return 0
    if command == "train":
        return run_train()
    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
