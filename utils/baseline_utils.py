from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, npz_path: Path):
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing train set: {npz_path}. Please run python data_preprocess/preprocess.py first.")
        data = np.load(npz_path, allow_pickle=True)
        self.x = np.asarray(data["X"], dtype=np.float32)
        self.y = np.asarray(data["y"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]).float().unsqueeze(0), torch.tensor(int(self.y[index]), dtype=torch.long)


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(name)


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def serializable_config(config: dict) -> dict:
    return {key: (value.as_posix() if isinstance(value, Path) else value) for key, value in config.items()}


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_config(payload), f, indent=2)


def open_csv_logger(path: Path, fieldnames: list[str]):
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer
