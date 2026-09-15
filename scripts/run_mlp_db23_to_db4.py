from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp_baseline.loaders import (
    NCTDataset,
    load_arrays_from_paths,
    seed_everything,
)
from mlp_baseline.model import WindowMLP


def loader_from_arrays(
    arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = NCTDataset(arrays, np.arange(len(arrays["labels"]), dtype=np.int64))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: WindowMLP,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float | int]:
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
    if count == 0:
        raise ValueError("数据集为空，无法计算结果。")
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "correct": correct,
        "samples": count,
    }


def evaluate_subjects(
    model: WindowMLP,
    subject_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for path in subject_paths:
        arrays = load_arrays_from_paths([path])
        metrics = run_epoch(
            model,
            loader_from_arrays(arrays, batch_size=batch_size, shuffle=False),
            device,
            optimizer=None,
        )
        results[path.stem] = metrics
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 DB2+DB3 训练简单 MLP，再在 DB4 上测试"
    )
    parser.add_argument("--config", type=Path, default=Path("config_mlp.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))

    data_root = Path(config["data_root"])
    train_paths = sorted(
        (data_root / "nct" / "train" / "db2").glob("subject_*.npz")
    ) + sorted((data_root / "nct" / "train" / "db3").glob("subject_*.npz"))
    test_paths = sorted((data_root / "nct" / "test" / "db4").glob("subject_*.npz"))
    if not train_paths:
        raise FileNotFoundError("没有找到 DB2/DB3 的训练 NCT 文件，请先运行 prepare_db23_to_db4.py。")
    if not test_paths:
        raise FileNotFoundError("没有找到 DB4 的测试 NCT 文件，请先运行 prepare_db23_to_db4.py。")

    train_arrays = load_arrays_from_paths(train_paths)
    train_loader = loader_from_arrays(train_arrays, args.batch_size, shuffle=True)
    first_batch = next(iter(train_loader))["windows"]
    _, _, channels, samples = first_batch.shape
    device_name = config.get("device", "cuda")
    device = torch.device(
        "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = WindowMLP(
        channels=channels,
        samples=samples,
        hidden_dim=args.hidden_dim,
        num_classes=int(config["num_classes"]),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(
        json.dumps(
            {
                "model": "MLP",
                "protocol": "DB2+DB3-train-DB4-test",
                "device": str(device),
                "train_subject_files": len(train_paths),
                "train_samples": len(train_loader.dataset),
                "test_subject_files": len(test_paths),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    final_train: dict[str, float | int] = {}
    for epoch in range(1, args.epochs + 1):
        final_train = run_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} train={final_train}", flush=True)

    subject_results = evaluate_subjects(model, test_paths, device, args.batch_size)
    total_correct = sum(int(metrics["correct"]) for metrics in subject_results.values())
    total_samples = sum(int(metrics["samples"]) for metrics in subject_results.values())
    overall = {
        "loss": sum(
            float(metrics["loss"]) * int(metrics["samples"])
            for metrics in subject_results.values()
        )
        / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }
    result = {
        "model": "window_flatten_mlp_masked_mean",
        "protocol": "DB2+DB3-train-DB4-test",
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "window_shape": [channels, samples],
        "train_subject_files": len(train_paths),
        "train_samples": len(train_loader.dataset),
        "final_train": final_train,
        "db4_overall": overall,
        "db4_subjects": subject_results,
    }
    output_dir = Path(config["results_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "db23_to_db4.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
