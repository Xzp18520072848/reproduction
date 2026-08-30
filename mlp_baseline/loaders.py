from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NCTDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "windows": torch.from_numpy(self.arrays["windows"][index]),
            "time_positions": torch.from_numpy(self.arrays["time_positions"][index]),
            "attention_mask": torch.from_numpy(self.arrays["attention_masks"][index]),
            "label": torch.tensor(self.arrays["labels"][index], dtype=torch.long),
            "subject": torch.tensor(self.arrays["subjects"][index], dtype=torch.long),
            "repetition": torch.tensor(self.arrays["repetitions"][index], dtype=torch.long),
        }


def load_subject(path: Path, subject: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
        return {
            "windows": np.asarray(data["windows"], dtype=np.float32),
            "time_positions": np.asarray(data["time_positions"], dtype=np.float32),
            "attention_masks": np.asarray(data["attention_masks"], dtype=np.bool_),
            "labels": labels,
            "subjects": np.full(len(labels), subject, dtype=np.int64),
            "repetitions": np.asarray(data["repetitions"], dtype=np.int64),
        }


def load_source_arrays(nct_dir: Path, held_out_subject: int) -> dict[str, np.ndarray]:
    paths = sorted(nct_dir.glob("subject_*.npz"))
    arrays_by_subject = []
    for path in paths:
        subject = int(path.stem.split("_")[-1])
        if subject != held_out_subject:
            arrays_by_subject.append(load_subject(path, subject))
    if not arrays_by_subject:
        raise ValueError("没有找到训练 subject。")
    return {
        key: np.concatenate([item[key] for item in arrays_by_subject], axis=0)
        for key in arrays_by_subject[0]
    }
