from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp_baseline.nct import tokenize_subject
from mlp_baseline.ninapro import load_ninapro_subject


def prepare_dataset(
    name: str,
    raw_root: Path,
    subject_ids: list[int],
    output_root: Path,
    config: dict,
) -> list[dict]:
    common = config["data"]
    split = "test" if name == "db4" else "train"
    records: list[dict] = []
    for subject_id in subject_ids:
        subject = load_ninapro_subject(
            root=raw_root,
            subject=int(subject_id),
            exercise=int(common["exercise"]),
            gesture_ids=[int(value) for value in common["gesture_ids"]],
            target_frequency=int(common["target_frequency"]),
            label_field=str(common["label_field"]),
            repetition_field=str(common["repetition_field"]),
        )
        labels = sorted(set(int(value) for value in subject.labels))
        expected = list(range(len(common["gesture_ids"])))
        if labels != expected:
            raise ValueError(
                f"{name} subject {subject_id} 的标签为 {labels}，"
                f"预期为 {expected}。停止处理，避免训练类别不一致。"
            )
        tokenized = tokenize_subject(
            subject_data=subject,
            window_ms=int(common["window_ms"]),
            step_ms=int(common["step_ms"]),
            max_sequence_length=int(common["max_sequence_length"]),
            threshold_method=str(common["threshold_method"]),
            threshold_percentile=float(common["threshold_percentile"]),
            keep_empty=False,
        )
        output_path = (
            output_root
            / "nct"
            / split
            / name
            / f"subject_{int(subject_id):03d}.npz"
        )
        tokenized.save(output_path)
        record = {
            "dataset": name,
            "subject": int(subject_id),
            "path": str(output_path),
            "samples": int(len(tokenized.labels)),
            "classes": labels,
            "window_shape": list(tokenized.windows.shape),
            "threshold": float(tokenized.threshold),
        }
        records.append(record)
        print(
            f"完成 {name} subject {subject_id}: "
            f"samples={record['samples']} -> {output_path}",
            flush=True,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 DB2+DB3 训练、DB4 测试的统一 NCT 数据"
    )
    parser.add_argument("--config", type=Path, default=Path("config_mlp.yaml"))
    parser.add_argument("--db2-root", type=Path, default=None)
    parser.add_argument("--db3-root", type=Path, default=None)
    parser.add_argument("--db4-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_roots = config["raw_roots"]
    subjects = config["subjects"]
    output_root = args.output_root or Path(config["data_root"])
    roots = {
        "db2": args.db2_root or Path(raw_roots["db2"]),
        "db3": args.db3_root or Path(raw_roots["db3"]),
        "db4": args.db4_root or Path(raw_roots["db4"]),
    }

    manifest: list[dict] = []
    for name in ("db2", "db3", "db4"):
        if not roots[name].is_dir():
            raise FileNotFoundError(
                f"{name} 原始数据目录不存在：{roots[name]}。"
                "请先解压官方 zip，或通过 --*-root 指定目录。"
            )
        manifest.extend(
            prepare_dataset(
                name=name,
                raw_root=roots[name],
                subject_ids=[int(value) for value in subjects[name]],
                output_root=output_root,
                config=config,
            )
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"全部完成：{len(manifest)} 个 subject，清单保存到 {manifest_path}")


if __name__ == "__main__":
    main()
