from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp_baseline.loaders import NCTDataset, load_source_arrays, load_subject, seed_everything


class WindowMLP(nn.Module):
    """对每个肌电窗口提取特征，再对一个样本里的窗口做平均。"""

    def __init__(self, channels: int, samples: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        input_dim = channels * samples
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, windows: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, samples = windows.shape
        flat = windows.reshape(batch_size * sequence_length, channels * samples)
        features = self.encoder(flat).reshape(batch_size, sequence_length, -1)
        mask = attention_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB4 简单 MLP 的 LOSO 实验")
    parser.add_argument("--config", type=Path, default=Path("config_mlp.yaml"))
    parser.add_argument("--held-out", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def loader_from_arrays(arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = NCTDataset(arrays, np.arange(len(arrays["labels"]), dtype=np.int64))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model: WindowMLP, loader: DataLoader, device: torch.device, optimizer: AdamW | None) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch in loader:
        windows = batch["windows"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(windows, mask)
        loss = F.cross_entropy(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        size = int(labels.numel())
        loss_sum += float(loss.detach()) * size
        correct += int((logits.argmax(-1) == labels).sum())
        count += size
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed_everything(int(config["seed"]))
    root = Path(config.get("legacy_data_root", "data/db4"))
    nct_dir = root / "nct"
    source = load_source_arrays(nct_dir, args.held_out)
    test = load_subject(nct_dir / f"subject_{args.held_out:03d}.npz", args.held_out)
    train_loader = loader_from_arrays(source, args.batch_size, True)
    test_loader = loader_from_arrays(test, args.batch_size, False)
    _, _, channels, samples = next(iter(train_loader))["windows"].shape
    device = torch.device("cuda" if config.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    model = WindowMLP(channels, samples, args.hidden_dim, int(config["num_classes"])).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(json.dumps({"model": "MLP", "device": str(device), "held_out": args.held_out, "source_samples": len(train_loader.dataset), "test_samples": len(test_loader.dataset)}), flush=True)
    final_train = {}
    for epoch in range(1, args.epochs + 1):
        final_train = run_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} train={final_train}", flush=True)
    final_test = run_epoch(model, test_loader, device, None)
    result = {
        "held_out": args.held_out,
        "model": "window_flatten_mlp_masked_mean",
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "window_shape": [channels, samples],
        "source_samples": len(train_loader.dataset),
        "test": final_test,
        "final_train": final_train,
        "protocol": "DB4-LOSO-supervised-MLP-over-NCT-windows",
    }
    output_dir = Path(config["results_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"fold_{args.held_out:03d}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
