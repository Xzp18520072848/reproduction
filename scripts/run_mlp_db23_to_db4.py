from __future__ import annotations

import argparse
import json
import random
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
    UnlabeledNCTDataset,
    load_arrays_from_paths,
    load_unlabeled_arrays_from_paths,
    seed_everything,
)
from mlp_baseline.model import MLPEncoder, MaskedTokenAutoencoder


def labeled_loader(
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


def unlabeled_loader(
    arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = UnlabeledNCTDataset(
        arrays, np.arange(len(arrays["windows"]), dtype=np.int64)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def sample_token_mask(attention_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 < ratio < 1.0:
        raise ValueError("mask_ratio 必须在 0 和 1 之间。")
    token_mask = (torch.rand(attention_mask.shape, device=attention_mask.device) < ratio)
    token_mask &= attention_mask.bool()
    valid_counts = attention_mask.sum(dim=1)
    masked_counts = token_mask.sum(dim=1)
    missing = (valid_counts > 0) & (masked_counts == 0)
    for row in torch.nonzero(missing, as_tuple=False).flatten().tolist():
        valid_positions = torch.nonzero(attention_mask[row], as_tuple=False).flatten()
        token_mask[row, valid_positions[0]] = True
    return token_mask


def ssl_epoch(
    model: MaskedTokenAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    mask_ratio: float,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    masked_tokens = 0
    for batch in loader:
        windows = batch["windows"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        time_positions = batch["time_positions"].to(device, non_blocking=True)
        token_mask = sample_token_mask(attention_mask, mask_ratio)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction, target = model.reconstruct(
            windows, attention_mask, time_positions, token_mask
        )
        selected = token_mask.reshape(-1)
        if not selected.any():
            continue
        loss = F.mse_loss(prediction.reshape(-1, prediction.shape[-1])[selected], target.reshape(-1, target.shape[-1])[selected])
        if training:
            loss.backward()
            optimizer.step()
        count = int(selected.sum())
        loss_sum += float(loss.detach()) * count
        masked_tokens += count
    if masked_tokens == 0:
        raise ValueError("没有可用于 SSL 重建的 masked token。")
    return {"loss": loss_sum / masked_tokens, "masked_tokens": masked_tokens}


class FrozenLinearProbe(torch.nn.Module):
    def __init__(self, encoder: MLPEncoder, channels: int, samples: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.channels = channels
        self.samples = samples
        self.head = torch.nn.Linear(encoder.hidden_dim, num_classes)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, windows: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, samples = windows.shape
        if channels != self.channels or samples != self.samples:
            raise ValueError(f"输入窗口维度为 {channels}x{samples}，与 encoder 不一致。")
        with torch.no_grad():
            flat = windows.reshape(batch_size * sequence_length, channels * samples)
            features = self.encoder(flat).reshape(batch_size, sequence_length, -1)
        mask = attention_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def classification_epoch(
    model: FrozenLinearProbe,
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
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(windows, attention_mask)
        loss = F.cross_entropy(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = int(labels.numel())
        loss_sum += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=-1) == labels).sum())
        count += batch_size
    if count == 0:
        raise ValueError("分类数据集为空。")
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "correct": correct,
        "samples": count,
    }


def split_pretrain_paths(paths: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * 0.1)))
    return shuffled[:-val_count], shuffled[-val_count:]


def run_pretraining(
    train_paths: list[Path],
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Path, dict]:
    ssl_config = config["ssl"]
    pretrain_train_paths, pretrain_val_paths = split_pretrain_paths(
        train_paths, int(config["seed"])
    )
    train_arrays = load_unlabeled_arrays_from_paths(pretrain_train_paths)
    val_arrays = load_unlabeled_arrays_from_paths(pretrain_val_paths)
    train_loader = unlabeled_loader(train_arrays, args.pretrain_batch_size, True)
    val_loader = unlabeled_loader(val_arrays, args.pretrain_batch_size, False)
    _, _, channels, samples = next(iter(train_loader))["windows"].shape
    model = MaskedTokenAutoencoder(channels, samples, args.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.pretrain_learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)
    mask_ratio = float(args.mask_ratio)
    checkpoint_path = Path(config["checkpoint_root"]) / "ssl_best_encoder.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    print(
        json.dumps(
            {
                "stage": "ssl_pretraining",
                "device": str(device),
                "train_subject_files": len(pretrain_train_paths),
                "val_subject_files": len(pretrain_val_paths),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "mask_ratio": mask_ratio,
                "uses_gesture_labels": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(1, args.pretrain_epochs + 1):
        train_metrics = ssl_epoch(model, train_loader, device, optimizer, mask_ratio)
        scheduler.step()
        torch.manual_seed(100000 + epoch)
        with torch.no_grad():
            val_metrics = ssl_epoch(model, val_loader, device, None, mask_ratio)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "channels": channels,
                    "samples": samples,
                    "hidden_dim": args.hidden_dim,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.pretrain_epochs:
            print(
                f"[SSL] epoch={epoch} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} best_epoch={best_epoch} "
                f"best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
    split_path = Path(config["checkpoint_root"]) / "pretrain_split.json"
    split_path.write_text(
        json.dumps(
            {
                "train_subject_files": [str(path) for path in pretrain_train_paths],
                "val_subject_files": [str(path) for path in pretrain_val_paths],
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "ratio": "9:1 by subject file",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not checkpoint_path.is_file():
        raise RuntimeError("SSL 预训练没有生成 best encoder checkpoint。")
    return checkpoint_path, {
        "train_subject_files": len(pretrain_train_paths),
        "val_subject_files": len(pretrain_val_paths),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": str(checkpoint_path),
        "mask_ratio": mask_ratio,
    }


def run_db4_fold(
    held_out: int,
    validation_subject: int,
    db4_paths: list[Path],
    checkpoint_path: Path,
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    fold_index: int,
) -> dict:
    test_paths = [path for path in db4_paths if path.stem == f"subject_{held_out:03d}"]
    val_paths = [path for path in db4_paths if path.stem == f"subject_{validation_subject:03d}"]
    train_paths = [path for path in db4_paths if path not in test_paths + val_paths]
    if len(test_paths) != 1 or len(val_paths) != 1 or len(train_paths) != 8:
        raise ValueError(
            f"LOSO 划分错误：held_out={held_out}, val={validation_subject}, "
            f"train_files={len(train_paths)}"
        )
    train_arrays = load_arrays_from_paths(train_paths)
    val_arrays = load_arrays_from_paths(val_paths)
    test_arrays = load_arrays_from_paths(test_paths)
    train_loader = labeled_loader(train_arrays, args.head_batch_size, True)
    val_loader = labeled_loader(val_arrays, args.head_batch_size, False)
    test_loader = labeled_loader(test_arrays, args.head_batch_size, False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder = MLPEncoder(
        int(checkpoint["channels"]),
        int(checkpoint["samples"]),
        int(checkpoint["hidden_dim"]),
    )
    encoder.load_state_dict(checkpoint["encoder"])
    model = FrozenLinearProbe(
        encoder,
        int(checkpoint["channels"]),
        int(checkpoint["samples"]),
        int(config["num_classes"]),
    ).to(device)
    optimizer = AdamW(model.head.parameters(), lr=args.head_learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.head_epochs)
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    head_checkpoint = Path(config["checkpoint_root"]) / "db4_loso" / f"fold_{held_out:03d}_best_head.pt"
    head_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[HEAD] fold={fold_index}/10 held_out={held_out} validation={validation_subject} "
        f"train_subjects=8 train_samples={len(train_loader.dataset)} "
        f"val_samples={len(val_loader.dataset)} test_samples={len(test_loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, args.head_epochs + 1):
        train_metrics = classification_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        val_metrics = classification_epoch(model, val_loader, device, None)
        if (
            float(val_metrics["accuracy"]) > best_val_accuracy
            or (
                float(val_metrics["accuracy"]) == best_val_accuracy
                and float(val_metrics["loss"]) < best_val_loss
            )
        ):
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "held_out": held_out,
                    "validation_subject": validation_subject,
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                    "val_loss": best_val_loss,
                },
                head_checkpoint,
            )
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.head_epochs:
            print(
                f"[HEAD] fold={held_out} epoch={epoch} "
                f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f} "
                f"best_epoch={best_epoch} best_val_acc={best_val_accuracy:.4f}",
                flush=True,
            )
    best_head = torch.load(head_checkpoint, map_location=device)
    model.head.load_state_dict(best_head["head"])
    test_metrics = classification_epoch(model, test_loader, device, None)
    result = {
        "held_out_subject": held_out,
        "validation_subject": validation_subject,
        "train_subjects": [int(path.stem.split("_")[-1]) for path in train_paths],
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "test": test_metrics,
        "best_head_checkpoint": str(head_checkpoint),
    }
    print(
        f"[HEAD] fold={held_out} BEST val_acc={best_val_accuracy:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DB2+DB3 无监督 SSL 预训练 + DB4 LOSO 线性分类"
    )
    parser.add_argument("--config", type=Path, default=Path("config_mlp.yaml"))
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--head-epochs", type=int, default=None)
    parser.add_argument("--pretrain-batch-size", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--pretrain-learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    return parser.parse_args()


def set_defaults_from_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    ssl_config = config["ssl"]
    values = {
        "pretrain_epochs": int(ssl_config["pretrain_epochs"]),
        "head_epochs": int(ssl_config["head_epochs"]),
        "pretrain_batch_size": int(ssl_config["pretrain_batch_size"]),
        "head_batch_size": int(ssl_config["head_batch_size"]),
        "hidden_dim": int(ssl_config["hidden_dim"]),
        "pretrain_learning_rate": float(ssl_config["pretrain_learning_rate"]),
        "head_learning_rate": float(ssl_config["head_learning_rate"]),
        "mask_ratio": float(ssl_config["mask_ratio"]),
        "print_every": int(ssl_config["print_every"]),
    }
    for key, value in values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.print_every <= 0:
        raise ValueError("print_every 必须为正整数。")
    return args


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args = set_defaults_from_config(args, config)
    seed_everything(int(config["seed"]))
    device_name = config.get("device", "cuda")
    device = torch.device(
        "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    )
    data_root = Path(config["data_root"])
    db2_paths = sorted((data_root / "nct" / "train" / "db2").glob("subject_*.npz"))
    db3_paths = sorted((data_root / "nct" / "train" / "db3").glob("subject_*.npz"))
    db4_paths = sorted((data_root / "nct" / "test" / "db4").glob("subject_*.npz"))
    if not db2_paths or not db3_paths or len(db4_paths) != 10:
        raise FileNotFoundError(
            f"数据文件数量不对：DB2={len(db2_paths)}, DB3={len(db3_paths)}, DB4={len(db4_paths)}"
        )
    checkpoint_path, pretrain_result = run_pretraining(
        db2_paths + db3_paths, config, args, device
    )
    db4_subject_ids = [int(path.stem.split("_")[-1]) for path in db4_paths]
    fold_results = []
    for index, held_out in enumerate(db4_subject_ids, start=1):
        validation_subject = db4_subject_ids[index % len(db4_subject_ids)]
        fold_results.append(
            run_db4_fold(
                held_out,
                validation_subject,
                db4_paths,
                checkpoint_path,
                config,
                args,
                device,
                index,
            )
        )
    total_correct = sum(int(item["test"]["correct"]) for item in fold_results)
    total_samples = sum(int(item["test"]["samples"]) for item in fold_results)
    summary = {
        "protocol": "DB2+DB3-SSL-pretrain-DB4-LOSO-linear-probe",
        "device": str(device),
        "pretraining": pretrain_result,
        "folds": len(fold_results),
        "mean_accuracy": float(np.mean([item["test"]["accuracy"] for item in fold_results])),
        "overall_accuracy": total_correct / total_samples,
        "total_test_samples": total_samples,
        "fold_results": fold_results,
    }
    output_dir = Path(config["results_root"]) / "ssl_db4_loso"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
