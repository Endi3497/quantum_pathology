
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import math

# Map text labels to binary targets; anything else is skipped.
STATUS_TO_LABEL = {"Positive": 1, "Negative": 0}


def _normalize_image_size(image_size: Sequence[int] | int) -> Tuple[int, int]:
    """Convert an int or tuple into a 2-tuple (H, W)."""
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a length-2 sequence")
    return (int(image_size[0]), int(image_size[1]))


def _load_label_table(csv_path: Path, target_column: str) -> Dict[str, int]:
    """
    Read the clinical CSV and keep only Positive/Negative rows.

    Returns a mapping of `<patient_id>.svs` -> int label.
    """
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(
            f"target_column must be one of {list(df.columns)}, got '{target_column}'"
        )

    label_series = df[["File Name", target_column]].copy()
    label_series[target_column] = label_series[target_column].map(STATUS_TO_LABEL)
    label_series = label_series.dropna(subset=[target_column])
    label_series[target_column] = label_series[target_column].astype(int)
    return dict(zip(label_series["File Name"], label_series[target_column]))


def _strip_suffix(path: Path) -> str:
    """Remove known suffixes (_2channel.npy or _2_channel.npy) to get patient id."""
    name = path.name
    for suffix in ("_2channel.npy", "_2_channel.npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    # Fallback to stem if suffix pattern is different.
    return path.stem


class TwoChannelGridDataset(Dataset):
    """
    Dataset for 2-channel numpy grid patches with binary ER/PR/HER2 labels.
    """

    def __init__(
        self,
        image_dir: Path | str,
        labels_csv: Path | str,
        target_column: str,
        image_size: Sequence[int] | int = 256,
        allowed_suffixes: Iterable[str] = ("_2channel.npy", "_2_channel.npy"),
        transform=None,
    ) -> None:
        """
        Args:
            image_dir: Directory containing 2-channel `.npy` files.
            labels_csv: CSV with `File Name` and target status columns.
            target_column: Label column name to learn.
            image_size: Desired output size (int or (H, W)).
            allowed_suffixes: File suffixes treated as 2-channel inputs.
            transform: Optional callable applied to the image tensor after
                resizing/normalization.
        """
        self.image_dir = Path(image_dir)
        self.labels_csv = Path(labels_csv)
        self.target_column = target_column
        self.image_size = _normalize_image_size(image_size)
        self.transform = transform
        self.allowed_suffixes = tuple(allowed_suffixes)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")
        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")

        self.label_map = _load_label_table(self.labels_csv, self.target_column)
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError("No images matched the label table with Positive/Negative values.")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for path in sorted(self.image_dir.iterdir()):
            if not any(str(path.name).endswith(suf) for suf in self.allowed_suffixes):
                continue
            patient_id = _strip_suffix(path)
            svs_name = f"{patient_id}.svs"
            label = self.label_map.get(svs_name)
            if label is None:
                continue
            samples.append((path, int(label)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(f"Expected array shape (H, W, 2), got {arr.shape} for {path}")
        tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        if tensor.shape[1:] != self.image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = self._load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path.name


def build_dataloader(
    image_dir: Path | str,
    labels_csv: Path | str,
    target_column: str,
    image_size: Sequence[int] | int = 256,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    transform=None,
) -> DataLoader:
    """
    Convenience helper that wires the dataset into a DataLoader.

    Returns:
        dataloader: torch.utils.data.DataLoader
        dataset: underlying TwoChannelGridDataset instance (for inspection)
    """
    dataset = TwoChannelGridDataset(
        image_dir=image_dir,
        labels_csv=labels_csv,
        target_column=target_column,
        image_size=image_size,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty; cannot split.")
    s = train_ratio + val_ratio + test_ratio
    if not math.isclose(s, 1.0, rel_tol=1e-3):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    test_len = total - train_len - val_len
    if test_len < 0:
        test_len = 0
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def build_loaders_with_split(
    image_dir: Path | str,
    labels_csv: Path | str,
    target_column: str,
    image_size: Sequence[int] | int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Tuple[DataLoader, Dataset], Tuple[DataLoader, Dataset], Tuple[DataLoader, Dataset]]:
    full_dataset = TwoChannelGridDataset(
        image_dir=image_dir,
        labels_csv=labels_csv,
        target_column=target_column,
        image_size=image_size,
    )
    train_set, val_set, test_set = split_dataset(full_dataset, train_ratio, val_ratio, test_ratio, seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return (train_loader, train_set), (val_loader, val_set), (test_loader, test_set)


def default_paths() -> Tuple[Path, Path]:
    """
    Provide the canonical image directory and CSV path for convenience.
    """
    base = Path("/home/junyeollee/QSANN/data/TCGA_BRCA_128x128")
    return (
        base / "grid" / "4patch",
        base / "DX1_ER_PR_HER2.csv",
    )
