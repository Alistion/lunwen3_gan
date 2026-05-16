# 文件路径: alistion/lunwen3_gan/train_mb_ddpm_lunwen3.py
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from timm.utils import ModelEmaV3

# 导入刚才添加的 mb-ddpm 模型
from models.mb_ddpm_1d import UNET_1D, DDPM_Scheduler_1D

CONFIG = {
    "data_dir": Path("processed/mhta_base_dataset"),
    "out_dir": Path("runs/mb_ddpm_lunwen3_v1"),
    "epochs": 2000,
    "batch_size": 32,
    "lr": 1e-4,
    "num_time_steps": 1000,
    "num_classes": 5,
    "ema_decay": 0.9999,
    "signal_length": 2048,
    "save_every": 500,
}

class SignalDataset(Dataset):
    def __init__(self, npz_path: Path):
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到数据集: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        self.x = np.asarray(data["X"], dtype=np.float32)
        self.y = np.asarray(data["y"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 输出形状 [1, 2048]
        return torch.from_numpy(self.x[index]).float().unsqueeze(0), torch.tensor(int(self.y[index]), dtype=torch.long)

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_mb_ddpm")

def main(config: dict = CONFIG) -> None:
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = config["out_dir"]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载你的振动信号数据
    dataset = SignalDataset(config["data_dir"] / "train.npz")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    # 2. 初始化 mb-ddpm 模型与调度器
    scheduler = DDPM_Scheduler_1D(num_time_steps=config["num_time_steps"]).to(device)
    model = UNET_1D(
        in_feature=config["signal_length"],
        time_steps=config["num_time_steps"],
        num_classes=config["num_classes"],
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    ema = ModelEmaV3(model, decay=config["ema_decay"])
    criterion = nn.MSELoss(reduction='mean')
    best_loss = float("inf")

    logger.info(f"开始训练 MB-DDPM，数据量: {len(dataset)}，设备: {device}")

    # 3. mb-ddpm 的训练循环
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"MB-DDPM epoch {epoch}", leave=False)
        
        for x, labels in pbar:
            x = x.to(device)
            labels = labels.to(device)
            batch_size = x.size(0)
            
            # 随机采样时间步 t
            t = torch.randint(0, config["num_time_steps"], (batch_size,), device=device)
            
            # 采样随机噪声
            e = torch.randn_like(x, device=device)
            
            # 取对应时间步的 alpha
            a = scheduler.alpha[t].view(batch_size, 1, 1).to(device)
            
            # 加噪
            x_t = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            
            # 预测噪声
            output = model(x_t, t, labels)
            
            optimizer.zero_grad()
            loss = criterion(output, e)
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch}/{config['epochs']} | Loss: {avg_loss:.6f}")

        # 4. 存盘点
        checkpoint = {
            'weights': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': int(epoch),
            'loss': float(avg_loss),
            'config': dict(config),
            'model_kwargs': {
                'in_feature': int(config["signal_length"]),
                'time_steps': int(config["num_time_steps"]),
                'num_classes': int(config["num_classes"]),
            },
        }

        if avg_loss < best_loss:
            best_loss = float(avg_loss)
            torch.save(checkpoint, ckpt_dir / "best_model.pt")

        torch.save(checkpoint, ckpt_dir / "latest_model.pt")

        if epoch % config["save_every"] == 0 or epoch == config["epochs"]:
            torch.save(checkpoint, ckpt_dir / f"mb_ddpm_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
