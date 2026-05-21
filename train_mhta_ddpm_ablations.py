from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_mhta_generated_metrics import CONFIG as EVAL_CONFIG
from evaluate_mhta_generated_metrics import main as evaluate_generated
from generate_mhta_ddpm import CONFIG as GENERATE_CONFIG
from generate_mhta_ddpm import main as generate_samples
from models.mhta_ddpm_1d import ConditionalMHTAUNet1D, DDPMScheduler1D
from train_mhta_ddpm import CONFIG as BASE_CONFIG
from train_mhta_ddpm import SignalDataset, get_device, model_kwargs, save_checkpoint, scheduler_kwargs, serializable_config
from utils.run_paths import create_run_dir, new_run_id
from utils.seed import set_seed


CONFIG = copy.deepcopy(BASE_CONFIG)
CONFIG.update(
    {
        "run_root": "runs/omc_tf_mhta_ddpm_v1",
        "ablation_root_name": "ablations",
        "run_id": None,
        "epochs": 3000,
        "generate_after_train": True,
        "evaluate_after_generate": True,
        "generated_subdir": "generated_dataset",
        "samples_per_fault_class": 100,
        "sampler": "ddpm",
        "num_inference_steps": 200,
        "eta": 0.0,
        "max_pairs_per_class": 100,
        "ci_bootstrap_repeats": 1000,
        "enabled_ablations": [
            "without_temporal_attention",
            "without_mhta_unet_only",
            "without_qkv_dwconv",
            "without_fft_loss",
            "input_kernel_3",
        ],
    }
)


ABLATION_CONFIGS = {
    "without_temporal_attention": {
        "description": "Remove bottleneck temporal self-attention.",
        "overrides": {
            "disable_temporal_attention": True,
        },
    },
    "without_mhta_unet_only": {
        "description": "Remove all MHTA and temporal attention blocks, keeping only the conditional U-Net backbone.",
        "overrides": {
            "disable_mhta": True,
            "disable_temporal_attention": True,
            "attention_levels": [],
        },
    },
    "without_qkv_dwconv": {
        "description": "Remove the depth-wise convolution after QKV projection in MHTA.",
        "overrides": {
            "use_qkv_dwconv": False,
        },
    },
    "without_fft_loss": {
        "description": "Train with DDPM noise prediction loss only.",
        "overrides": {
            "lambda_fft": 0.0,
        },
    },
    "input_kernel_3": {
        "description": "Use a 3-point input convolution instead of the large 31-point input convolution.",
        "overrides": {
            "input_kernel_size": 3,
        },
    },
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_mhta_ddpm_ablations")


def apply_overrides(base_config: dict, ablation_name: str, batch_run_id: str) -> dict:
    if ablation_name not in ABLATION_CONFIGS:
        raise KeyError(f"Unknown ablation: {ablation_name}")
    config = copy.deepcopy(base_config)
    config.update(ABLATION_CONFIGS[ablation_name]["overrides"])
    config["ablation_name"] = ablation_name
    config["ablation_description"] = ABLATION_CONFIGS[ablation_name]["description"]
    config["run_id"] = batch_run_id
    return config


def compute_loss(
    pred_noise: torch.Tensor,
    noise: torch.Tensor,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler1D,
    config: dict,
) -> torch.Tensor:
    loss_noise = F.mse_loss(pred_noise, noise)
    lambda_fft = float(config.get("lambda_fft", 0.05))
    if bool(config.get("strict_paper_mode", False)) or lambda_fft <= 0.0:
        return loss_noise

    alpha_bar_t = scheduler._extract(scheduler.alphas_cumprod, timesteps, x_t.shape)
    pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    pred_x0 = torch.clamp(pred_x0, -25.0, 25.0)

    fft_real = torch.fft.rfft(x0, dim=-1, norm="ortho")
    fft_pred = torch.fft.rfft(pred_x0, dim=-1, norm="ortho")
    mag_real = torch.abs(fft_real)
    mag_pred = torch.abs(fft_pred)
    loss_fft = F.mse_loss(mag_pred[:, :, :300], mag_real[:, :, :300])
    return loss_noise + lambda_fft * loss_fft


def train_one_ablation(config: dict, logger: logging.Logger) -> dict:
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    ablation_name = str(config["ablation_name"])
    run_root = Path(config["run_root"]) / str(config["ablation_root_name"]) / ablation_name
    out_dir = create_run_dir(run_root, str(config["run_id"]))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_config(config), f, indent=2)

    dataset = SignalDataset(Path(config["data_dir"]) / "train.npz")
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = ConditionalMHTAUNet1D(**model_kwargs(config)).to(device)
    scheduler = DDPMScheduler1D(**scheduler_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    log_path = out_dir / "train_log.csv"
    best_loss = float("inf")
    logger.info("Training ablation=%s on %d samples with device=%s", ablation_name, len(dataset), device)
    logger.info("Run directory: %s", out_dir)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "best_loss"])
        writer.writeheader()
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            losses = []
            pbar = tqdm(dataloader, desc=f"{ablation_name} epoch {epoch:03d}", leave=False)
            for x0, labels in pbar:
                x0 = x0.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                noise = torch.randn_like(x0)
                timesteps = torch.randint(
                    0,
                    int(config["num_train_timesteps"]),
                    (x0.size(0),),
                    dtype=torch.long,
                    device=device,
                )
                x_t = scheduler.q_sample(x0, timesteps, noise)
                pred_noise = model(x_t, timesteps, labels)
                loss = compute_loss(pred_noise, noise, x0, x_t, timesteps, scheduler, config)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()

                loss_value = float(loss.item())
                losses.append(loss_value)
                pbar.set_postfix(loss=f"{loss_value:.4f}")

            epoch_loss = float(np.mean(losses)) if losses else float("nan")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, epoch_loss, config)
            save_checkpoint(ckpt_dir / "latest_model.pt", model, optimizer, epoch, epoch_loss, config)
            if epoch % int(config["save_every"]) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, epoch_loss, config)

            writer.writerow({"epoch": epoch, "loss": epoch_loss, "best_loss": best_loss})
            f.flush()
            logger.info("ablation=%s epoch=%03d loss=%.6f best=%.6f", ablation_name, epoch, epoch_loss, best_loss)

    return {
        "ablation_name": ablation_name,
        "description": str(config["ablation_description"]),
        "run_dir": out_dir.as_posix(),
        "best_loss": best_loss,
        "epochs": int(config["epochs"]),
    }


def run_generation(config: dict, run_dir: str, logger: logging.Logger) -> None:
    ablation_name = str(config["ablation_name"])
    gen_config = dict(GENERATE_CONFIG)
    gen_config.update(
        {
            "data_dir": Path(config["data_dir"]),
            "run_root": Path(run_dir).parent,
            "run_id": Path(run_dir).name,
            "ckpt": "best_model.pt",
            "generated_subdir": str(config.get("generated_subdir", "generated_dataset")),
            "samples_per_fault_class": int(config.get("samples_per_fault_class", 100)),
            "batch_size": int(config["batch_size"]),
            "signal_length": int(config["signal_length"]),
            "sampler": str(config.get("sampler", "ddpm")),
            "num_inference_steps": int(config.get("num_inference_steps", 200)),
            "eta": float(config.get("eta", 0.0)),
            "seed": int(config["seed"]),
            "device": str(config["device"]),
        }
    )
    logger.info("Generating samples for ablation=%s", ablation_name)
    generate_samples(gen_config)


def run_evaluation(config: dict, run_dir: str, logger: logging.Logger) -> None:
    ablation_name = str(config["ablation_name"])
    eval_config = dict(EVAL_CONFIG)
    eval_config.update(
        {
            "real_npz": Path(config["data_dir"]) / "train.npz",
            "run_root": Path(run_dir).parent,
            "run_id": Path(run_dir).name,
            "generated_subdir": str(config.get("generated_subdir", "generated_dataset")),
            "seed": int(config["seed"]),
            "max_pairs_per_class": int(config.get("max_pairs_per_class", 100)),
            "ci_bootstrap_repeats": int(config.get("ci_bootstrap_repeats", 1000)),
        }
    )
    logger.info("Evaluating generated samples for ablation=%s", ablation_name)
    evaluate_generated(eval_config)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_metric_rows(train_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    pair_rows = []
    class_rows = []
    overall_rows = []
    for train_row in train_rows:
        ablation_name = str(train_row["ablation_name"])
        run_dir = Path(str(train_row["run_dir"]))
        pair_path = run_dir / "evaluation" / "mhta_pair_metrics.csv"
        quality_path = run_dir / "evaluation" / "mhta_generated_quality_metrics.csv"
        overall_path = run_dir / "evaluation" / "mhta_overall_metrics.csv"

        for row in read_csv_rows(pair_path):
            row = dict(row)
            row["ablation_name"] = ablation_name
            row["run_dir"] = run_dir.as_posix()
            pair_rows.append(row)

        for row in read_csv_rows(quality_path):
            row = dict(row)
            row["ablation_name"] = ablation_name
            row["run_dir"] = run_dir.as_posix()
            class_rows.append(row)

        for row in read_csv_rows(overall_path):
            row = dict(row)
            row["ablation_name"] = ablation_name
            row["run_dir"] = run_dir.as_posix()
            row["best_loss"] = str(train_row["best_loss"])
            overall_rows.append(row)

    return pair_rows, class_rows, overall_rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MHTA-DDPM ablation variants.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs for every ablation.")
    parser.add_argument("--run-id", type=str, default=None, help="Use a fixed run id under each ablation folder.")
    parser.add_argument("--skip-generate", action="store_true", help="Only train ablations; do not generate samples.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Generate samples but skip metric evaluation.")
    parser.add_argument("--samples-per-fault-class", type=int, default=None, help="Override generated samples per fault class.")
    parser.add_argument("--max-pairs-per-class", type=int, default=None, help="Override evaluation pairs per class.")
    parser.add_argument("--ci-bootstrap-repeats", type=int, default=None, help="Override bootstrap repeats for CI std.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(ABLATION_CONFIGS),
        default=None,
        help="Train only the selected ablations.",
    )
    return parser.parse_args()


def main(config: dict = CONFIG) -> None:
    args = parse_args()
    logger = setup_logger()
    base_config = copy.deepcopy(config)
    if args.epochs is not None:
        base_config["epochs"] = int(args.epochs)
    if args.skip_generate:
        base_config["generate_after_train"] = False
    if args.skip_evaluate:
        base_config["evaluate_after_generate"] = False
    if args.samples_per_fault_class is not None:
        base_config["samples_per_fault_class"] = int(args.samples_per_fault_class)
    if args.max_pairs_per_class is not None:
        base_config["max_pairs_per_class"] = int(args.max_pairs_per_class)
    if args.ci_bootstrap_repeats is not None:
        base_config["ci_bootstrap_repeats"] = int(args.ci_bootstrap_repeats)
    if args.only is not None:
        base_config["enabled_ablations"] = list(args.only)

    batch_run_id = str(args.run_id or base_config.get("run_id") or new_run_id())
    summary_rows = []
    for ablation_name in base_config["enabled_ablations"]:
        ablation_config = apply_overrides(base_config, str(ablation_name), batch_run_id)
        train_row = train_one_ablation(ablation_config, logger)
        summary_rows.append(train_row)
        if bool(base_config.get("generate_after_train", True)):
            run_generation(ablation_config, str(train_row["run_dir"]), logger)
        if bool(base_config.get("generate_after_train", True)) and bool(base_config.get("evaluate_after_generate", True)):
            run_evaluation(ablation_config, str(train_row["run_dir"]), logger)

    summary_dir = Path(base_config["run_root"]) / str(base_config["ablation_root_name"])
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"ablation_summary_{batch_run_id}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation_name", "description", "run_dir", "best_loss", "epochs"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Saved ablation summary to %s", summary_path)

    pair_rows, class_rows, overall_rows = collect_metric_rows(summary_rows)
    pair_summary_path = summary_dir / f"ablation_pair_metrics_summary_{batch_run_id}.csv"
    class_summary_path = summary_dir / f"ablation_class_metrics_summary_{batch_run_id}.csv"
    overall_summary_path = summary_dir / f"ablation_metrics_summary_{batch_run_id}.csv"
    write_rows(pair_summary_path, pair_rows)
    write_rows(class_summary_path, class_rows)
    write_rows(overall_summary_path, overall_rows)
    if pair_rows:
        logger.info("Saved pair metric summary to %s", pair_summary_path)
    if class_rows:
        logger.info("Saved class metric summary to %s", class_summary_path)
    if overall_rows:
        logger.info("Saved overall metric summary to %s", overall_summary_path)


if __name__ == "__main__":
    main()
