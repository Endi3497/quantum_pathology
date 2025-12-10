from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import os

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
    parser.add_argument("--q-dim", type=int, default=32)
    parser.add_argument("--k-dim", type=int, default=32)
    parser.add_argument("--v-dim", type=int, default=32)

    # Data split
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--no-patient-split",
        action="store_false",
        dest="patient_split",
        help="Disable patient/slide-level split (default: enabled).",
    )
    parser.set_defaults(patient_split=True)

    # Attention
    parser.add_argument("--attn-type", type=str, default="dot", choices=["dot", "rbf"])
    parser.add_argument("--attn-layers", type=int, default=1)
    parser.add_argument("--rbf-gamma", type=float, default=1.0)

    # Aggregation
    parser.add_argument("--agg-mode", type=str, default="concat", choices=["concat", "gap_gmp"])

    # Classifier
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[128])
    parser.add_argument("--dropout", type=float, default=0.5)

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
    # Grid search options (provide multiple values to sweep)
    parser.add_argument("--grid-image-sizes", type=int, nargs="*", help="List of image sizes to sweep.")
    parser.add_argument("--grid-patch-sizes", type=int, nargs="*", help="List of patch sizes to sweep.")
    parser.add_argument("--grid-vqc-layers", type=int, nargs="*", help="List of vqc_layers to sweep.")
    parser.add_argument("--grid-measurements", type=str, nargs="*", choices=["statevector", "correlations"])
    parser.add_argument("--grid-qkv-modes", type=str, nargs="*", choices=["shared", "separate"])
    parser.add_argument("--grid-q-dims", type=int, nargs="*", help="List of q dims to sweep.")
    parser.add_argument("--grid-k-dims", type=int, nargs="*", help="List of k dims to sweep.")
    parser.add_argument("--grid-v-dims", type=int, nargs="*", help="List of v dims to sweep.")
    parser.add_argument("--grid-attn-layers", type=int, nargs="*", help="List of attention layers to sweep.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    best_overall = {"auroc": -1.0, "run_name": None}

    def run_once(cfg: argparse.Namespace) -> dict:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        (train_loader, train_set), (val_loader, val_set), (test_loader, test_set) = build_loaders_with_split(
            image_dir=cfg.image_dir,
            labels_csv=cfg.labels_csv,
            target_column=cfg.target_column,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True if "cuda" in cfg.device else False,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            patient_split=cfg.patient_split,
        )
        print(
            f"[run {cfg.run_name}] Data splits -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)} | "
            f"batch_size: {cfg.batch_size}, image_size: {cfg.image_size}, patch_size: {cfg.patch_size}, device: {device}"
        )

        ansatz = QuantumAnsatz(
            data_dim=2 * cfg.patch_size * cfg.patch_size,
            num_qubits=cfg.num_qubits,
            vqc_layers=cfg.vqc_layers,
            measurement=cfg.measurement,
            backend_device=cfg.backend_device,
            use_torch_autograd=cfg.use_torch_autograd,
        )
        if cfg.qkv_mode == "shared":
            model = HybridQuantumClassifier(
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                ansatz=ansatz,
                q_dim=cfg.q_dim,
                k_dim=cfg.k_dim,
                v_dim=cfg.v_dim,
                attn_layers=cfg.attn_layers,
                attn_type=cfg.attn_type,
                rbf_gamma=cfg.rbf_gamma,
                agg_mode=cfg.agg_mode,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
                device=device,
                qkv_mode="shared",
            )
        else:
            ansatz_k = QuantumAnsatz(
                data_dim=2 * cfg.patch_size * cfg.patch_size,
                num_qubits=cfg.num_qubits,
                vqc_layers=cfg.vqc_layers,
                measurement=cfg.measurement,
                backend_device=cfg.backend_device,
                use_torch_autograd=cfg.use_torch_autograd,
            )
            ansatz_v = QuantumAnsatz(
                data_dim=2 * cfg.patch_size * cfg.patch_size,
                num_qubits=cfg.num_qubits,
                vqc_layers=cfg.vqc_layers,
                measurement=cfg.measurement,
                backend_device=cfg.backend_device,
                use_torch_autograd=cfg.use_torch_autograd,
            )
            model = HybridQuantumClassifier(
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                ansatz=ansatz,
                ansatz_k=ansatz_k,
                ansatz_v=ansatz_v,
                q_dim=cfg.q_dim,
                k_dim=cfg.k_dim,
                v_dim=cfg.v_dim,
                attn_layers=cfg.attn_layers,
                attn_type=cfg.attn_type,
                rbf_gamma=cfg.rbf_gamma,
                agg_mode=cfg.agg_mode,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
                device=device,
                qkv_mode="separate",
            )
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

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

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = run_epoch(train_loader, train=True)
            val_loss, val_acc = run_epoch(val_loader, train=False)
            print(
                f"[{cfg.run_name}] Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if cfg.early_stop:
                if val_loss + cfg.early_stop_min_delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stop_patience:
                        print(f"[{cfg.run_name}] Early stopping triggered.")
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
        metrics = {}
        if all_labels:
            y_true = torch.cat(all_labels).numpy()
            y_scores = torch.cat(all_outputs).numpy()
            y_pred = (y_scores >= 0.5).astype(int)
            metrics["acc"] = accuracy_score(y_true, y_pred)
            try:
                metrics["auroc"] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics["auroc"] = float("nan")
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            metrics.update({"precision": precision, "recall": recall, "f1": f1})
            print(
                f"[{cfg.run_name}] Test -> acc: {metrics['acc']:.4f} auroc: {metrics['auroc']:.4f} "
                f"precision: {metrics['precision']:.4f} recall: {metrics['recall']:.4f} f1: {metrics['f1']:.4f}"
            )
            if not (metrics["auroc"] != metrics["auroc"]):  # not NaN
                if metrics["auroc"] > best_overall["auroc"]:
                    best_overall["auroc"] = metrics["auroc"]
                    best_overall["run_name"] = cfg.run_name
                    os.makedirs(cfg.model_dir, exist_ok=True)
                    save_path = Path(cfg.model_dir) / f"{cfg.run_name}_best.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"[best] Updated best AUROC: {metrics['auroc']:.4f} | saved model to {save_path}")
        return metrics

    # Grid search handling
    grid_fields = {
        "image_size": args.grid_image_sizes,
        "patch_size": args.grid_patch_sizes,
        "vqc_layers": args.grid_vqc_layers,
        "measurement": args.grid_measurements,
        "qkv_mode": args.grid_qkv_modes,
        "q_dim": args.grid_q_dims,
        "k_dim": args.grid_k_dims,
        "v_dim": args.grid_v_dims,
        "attn_layers": args.grid_attn_layers,
    }
    sweep_lists = {k: v for k, v in grid_fields.items() if v}

    if not sweep_lists:
        run_once(args)
    else:
        keys, values = zip(*sweep_lists.items())
        for combo in itertools.product(*values):
            cfg = copy.deepcopy(args)
            for k, v in zip(keys, combo):
                setattr(cfg, k, v)
            cfg.run_name = f"grid_" + "_".join(f"{k}={v}" for k, v in zip(keys, combo))
            run_once(cfg)
            if best_overall["run_name"]:
                print(f"[progress] Current best run: {best_overall['run_name']} | AUROC={best_overall['auroc']:.4f}")


if __name__ == "__main__":
    main()
