"""Training CLI skeleton for AdaptiveGuard."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaptiveGuard risk model")
    parser.add_argument("--data-path", required=False, help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Starting training placeholder... "
        f"data_path={args.data_path!r}, epochs={args.epochs}"
    )
    print("Training pipeline not implemented yet.")


if __name__ == "__main__":
    main()
