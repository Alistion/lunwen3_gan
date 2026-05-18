from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.baseline_ddpm_1d import ConditionalUNet1D
from models.mhta_ddpm_1d import DDPMScheduler1D
from utils.baseline_utils import get_device, setup_logger
from utils.run_paths import resolve_existing_run_dir
from utils.seed import set_seed
from utils.signal_utils import CLASS_NAMES, LABEL_MAP, order_template_for_labels

CONFIG = {
    "run_root": "runs/baseline_ddpm_v1",
    "run_id": None,
    "ckpt": None,
    "generated_subdir": "generated_dataset",
    "samples_per_fault_class": 100,
    "batch_size": 32,
    "signal_length": 2048,
    "sampler": "ddim",
    "num_inference_steps": 200,
    "eta": 0.0,
    "seed": 42,
    "device": "cuda_if_available",
}


def resolve_checkpoint_path(config: dict) -> Path:
    explicit_ckpt = config.get("ckpt")
    run_id = config.get("run_id")
    if not explicit_ckpt and isinstance(run_id, str) and run_id.endswith(".pt"):
        explicit_ckpt = run_id
        run_id = None
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), run_id)
    if explicit_ckpt:
        ckpt_path = Path(explicit_ckpt)
        if ckpt_path.is_absolute() or ckpt_path.parent != Path("."):
            return ckpt_path
        return run_dir / "checkpoints" / ckpt_path.name
    return run_dir / "checkpoints" / "best_model.pt"


def main(config: dict = CONFIG) -> None:
    logger = setup_logger("generate_baseline_ddpm")
    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    run_id = config.get("run_id")
    if not config.get("ckpt") and isinstance(run_id, str) and run_id.endswith(".pt"):
        run_id = None
    run_dir = resolve_existing_run_dir(Path(config["run_root"]), run_id)
    out_dir = run_dir / str(config["generated_subdir"]); out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(resolve_checkpoint_path(config), map_location=device)
    model = ConditionalUNet1D(**payload["model_kwargs"]).to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
    scheduler = DDPMScheduler1D(**payload["scheduler_kwargs"]).to(device)
    xs, ys = [], []
    with torch.no_grad():
        for class_name in CLASS_NAMES[1:]:
            label = LABEL_MAP[class_name]
            left = int(config["samples_per_fault_class"])
            for _ in tqdm(range((left + int(config["batch_size"]) - 1) // int(config["batch_size"])), desc=class_name, leave=False):
                bsz = min(int(config["batch_size"]), left)
                labels = torch.full((bsz,), label, dtype=torch.long, device=device)
                shape = (bsz, 1, int(config["signal_length"]))
                fake = scheduler.ddim_sample(model, shape, labels, int(config["num_inference_steps"]), float(config["eta"]), device) if str(config["sampler"]).lower() == "ddim" else scheduler.sample(model, shape, labels, device)
                xs.append(fake.squeeze(1).cpu().numpy().astype(np.float32)); ys.append(np.full(bsz, label, dtype=np.int64))
                left -= bsz
    x_gen, y_gen = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(out_dir / "generated_only.npz", X=x_gen, y=y_gen, class_names=np.asarray(CLASS_NAMES), order_templates=order_template_for_labels(y_gen))
    logger.info("Saved baseline DDPM generated samples to %s", out_dir)


if __name__ == "__main__":
    main()
