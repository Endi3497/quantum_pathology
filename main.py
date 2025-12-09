from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import build_loaders_with_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum patch model training entrypoint")

    # Data
    parser.add_argument("--image-dir", type=Path, default=Path("data/TCGA_BRCA_128x128/grid/4patch"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/TCGA_BRCA_128x128/DX1_ER_PR_HER2.csv"))
    parser.add_argument("--target-column", type=str, default="ER_status_BCR", choices=["ER_status_BCR", "PR_status_BCR", "HER2_status_BCR"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=4)

    # Quantum ansatz
    parser.add_argument("--num-qubits", type=int, default=8)
    parser.add_argument("--vqc-layers", type=int, default=1)
    parser.add_argument("--measurement", type=str, default="statevector", choices=["statevector", "correlations"])
    parser.add_argument("--backend-device", type=str, default="cpu", choices=["cpu", "gpu"])

    # QKV options
    parser.add_argument("--qkv-mode", type=str, default="shared", choices=["shared", "separate"])
    parser.add_argument("--q-dim", type=int, default=64)
    parser.add_argument("--k-dim", type=int, default=64)
    parser.add_argument("--v-dim", type=int, default=128)

    # Data split
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    # Attention
    parser.add_argument("--attn-type", type=str, default="dot", choices=["dot", "rbf"])
    parser.add_argument("--attn-layers", type=int, default=1)
    parser.add_argument("--rbf-gamma", type=float, default=1.0)

    # Aggregation
    parser.add_argument("--agg-mode", type=str, default="gap_gmp", choices=["concat", "gap_gmp"])

    # Classifier
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[128])
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    # Logging / checkpoints
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--run-name", type=str, default="run")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    (train_loader, train_set), (val_loader, val_set), (test_loader, test_set) = build_loaders_with_split(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        target_column=args.target_column,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(
        f"Data splits -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)} | "
        f"batch_size: {args.batch_size}, image_size: {args.image_size}"
    )
    # TODO: add training loop using early-stop options when model wiring is complete.


if __name__ == "__main__":
    main()
