from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp_baseline.db4 import load_db4_subject
from mlp_baseline.nct import tokenize_subject


def save_raw(subject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        subject=subject.subject,
        signals=subject.signals,
        lengths=subject.lengths,
        labels=subject.labels,
        repetitions=subject.repetitions,
        rest_signals=subject.rest_signals,
        rest_lengths=subject.rest_lengths,
        frequency=subject.frequency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="把 DB4 原始 mat 文件处理成 MLP 可以读取的 NCT 数据")
    parser.add_argument("--config", type=Path, default=Path("config_mlp.yaml"))
    parser.add_argument("--raw-root", type=Path, required=True, help="DB4 原始数据目录，里面应有 s1、s2 等文件夹")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["data_root"])
    common = config["data"]
    output_root.mkdir(parents=True, exist_ok=True)

    for subject_id in config["subject_ids"]:
        subject = load_db4_subject(
            root=args.raw_root,
            subject=int(subject_id),
            exercise=int(common["exercise"]),
            gesture_ids=common["gesture_ids"],
            target_frequency=int(common["target_frequency"]),
            label_field=str(common["label_field"]),
            repetition_field=str(common["repetition_field"]),
        )
        raw_path = output_root / "raw" / f"subject_{int(subject_id):03d}.npz"
        save_raw(subject, raw_path)
        tokenized = tokenize_subject(
            subject_data=subject,
            window_ms=int(common["window_ms"]),
            step_ms=int(common["step_ms"]),
            max_sequence_length=int(common["max_sequence_length"]),
            threshold_method=str(common["threshold_method"]),
            threshold_percentile=float(common["threshold_percentile"]),
            keep_empty=False,
        )
        nct_path = output_root / "nct" / f"subject_{int(subject_id):03d}.npz"
        tokenized.save(nct_path)
        print(f"完成 subject {subject_id}: {nct_path}")


if __name__ == "__main__":
    main()
