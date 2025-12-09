from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from data_loader import build_loaders_with_split
from model import (
    HybridQuantumClassifier,
    QuantumAnsatz,
    QuantumPatchModel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum patch model training entrypoint")

    # Data
    parser.add_argument("--image-dir", type=Path, default=Path("/home/junyeollee/QSANN/data/TCGA_BRCA_128x128/grid/4patch"))
    parser.add_argument("--labels-csv", type=Path, default=Path("/home/junyeollee/QSANN/data/TCGA_BRCA_128x128/DX1_ER_PR_HER2.csv"))
    parser.add_argument("--target-column", type=str, default="ER_status_BCR", choices=["ER_status_BCR", "PR_status_BCR", "HER2_status_BCR"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)

    # Quantum ansatz
    parser.add_argument("--num-qubits", type=int, default=8)
    parser.add_argument("--vqc-layers", type=int, default=1)
    parser.add_argument("--measurement", type=str, default="correlations", choices=["statevector", "correlations"])
    parser.add_argument("--backend-device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--use-torch-autograd", action="store_true", default=True)

    # QKV options
    parser.add_argument("--qkv-mode", type=str, default="shared", choices=["shared", "separate"])
    parser.add_argument("--q-dim", type=int, default=64)
    parser.add_argument("--k-dim", type=int, default=64)
    parser.add_argument("--v-dim", type=int, default=64)

    # Data split
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    # Attention
    parser.add_argument("--attn-type", type=str, default="dot", choices=["dot", "rbf"])
    parser.add_argument("--attn-layers", type=int, default=1)
    parser.add_argument("--rbf-gamma", type=float, default=1.0)

    # Aggregation
    parser.add_argument("--agg-mode", type=str, default="concat", choices=["concat", "gap_gmp"])

    # Classifier
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[128])
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    # Logging / checkpoints
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--run-name", type=str, default="test")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    (train_loader, train_set), (val_loader, val_set), (test_loader, test_set) = build_loaders_with_split(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        target_column=args.target_column,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda:0" else False,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(
        f"Data splits -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)} | "
        f"batch_size: {args.batch_size}, image_size: {args.image_size}, device: {device}"
    )

    ansatz = QuantumAnsatz(
        data_dim=2 * args.patch_size * args.patch_size,
        num_qubits=args.num_qubits,
        vqc_layers=args.vqc_layers,
        measurement=args.measurement,
        backend_device=args.backend_device,
        use_torch_autograd=args.use_torch_autograd,
    )
    model = HybridQuantumClassifier(
        image_size=args.image_size,
        patch_size=args.patch_size,
        ansatz=ansatz,
        q_dim=args.q_dim,
        k_dim=args.k_dim,
        v_dim=args.v_dim,
        attn_layers=args.attn_layers,
        attn_type=args.attn_type,
        rbf_gamma=args.rbf_gamma,
        agg_mode=args.agg_mode,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        device=device,
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    def run_epoch(loader, train: bool):
        nonlocal best_val_loss, patience_counter
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total = 0
        all_labels = []
        all_outputs = []
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_labels.append(labels.detach().cpu())
            all_outputs.append(outputs.detach().cpu())
        if all_labels:
            labels_cat = torch.cat(all_labels)
            outputs_cat = torch.cat(all_outputs)
            preds = (outputs_cat >= 0.5).int()
            acc = accuracy_score(labels_cat.numpy(), preds.numpy())
        else:
            acc = 0.0
        return total_loss / max(1, total), acc

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if args.early_stop:
            if val_loss + args.early_stop_min_delta < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print("Early stopping triggered.")
                    break

    # Test evaluation
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())
    if all_labels:
        y_true = torch.cat(all_labels).numpy()
        y_scores = torch.cat(all_outputs).numpy()
        y_pred = (y_scores >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        print(
            f"Test metrics -> acc: {acc:.4f} auroc: {auroc:.4f} "
            f"precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}"
        )


if __name__ == "__main__":
    main()
