from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NPZSignalDataset
from models.wdcnn import WDCNN
from utils.plotting import plot_confusion_matrix, plot_training_curve
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES

CONFIG = {
    "baseline_data_dir": Path("processed/omc_tf_gan_dataset"),
    "augmented_data_dir": Path("processed/omc_tf_gan_dataset_augmented"),
    "out_dir": Path("runs/wdcnn_compare_v2"),
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    "seed": 42,
    "device": "cuda_if_available",
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_wdcnn_compare")


def get_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    return total_loss / max(total, 1), true, pred


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    row = {
        "experiment": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    recalls = recall_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average=None, zero_division=0)
    for class_name, value in zip(CLASS_NAMES, recalls):
        row[f"Recall_{class_name}"] = value
    return pd.DataFrame([row])


def train_experiment(name: str, train_npz: Path, val_npz: Path, test_npz: Path, out_dir: Path, epochs: int, batch_size: int, lr: float, device: torch.device) -> None:
    train_loader = DataLoader(NPZSignalDataset(train_npz, include_conditions=False), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(NPZSignalDataset(val_npz, include_conditions=False), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(NPZSignalDataset(test_npz, include_conditions=False), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = WDCNN(num_classes=len(CLASS_NAMES), in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_true, val_pred = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        logging.getLogger("train_wdcnn_compare").info("%s epoch %03d train_acc=%.4f val_f1=%.4f", name, epoch, train_acc, val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    suffix = "baseline" if name == "baseline" else "gan_augmented"
    metrics_row(name, test_true, test_pred).to_csv(out_dir / f"{suffix}_metrics.csv", index=False)
    plot_confusion_matrix(confusion_matrix(test_true, test_pred, labels=list(range(len(CLASS_NAMES)))), out_dir / f"confusion_matrix_{suffix}.png", f"{name} confusion matrix")
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / f"training_curve_{suffix}.csv", index=False)
    plot_training_curve(hist, out_dir / f"training_curve_{suffix}.png", f"{name} training curve")


def main(config: dict = CONFIG) -> None:
    setup_logger()
    set_seed(int(config["seed"]))
    baseline_dir = config["baseline_data_dir"]
    augmented_dir = config["augmented_data_dir"]
    out_dir = config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(config["device"])
    train_experiment(
        "baseline",
        baseline_dir / "train.npz",
        baseline_dir / "val.npz",
        baseline_dir / "test.npz",
        out_dir,
        config["epochs"],
        config["batch_size"],
        config["lr"],
        device,
    )
    train_experiment(
        "gan_augmented",
        augmented_dir / "train_augmented.npz",
        baseline_dir / "val.npz",
        baseline_dir / "test.npz",
        out_dir,
        config["epochs"],
        config["batch_size"],
        config["lr"],
        device,
    )


if __name__ == "__main__":
    main()
