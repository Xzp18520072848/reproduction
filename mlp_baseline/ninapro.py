from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from .data import CachedSubject, contiguous_runs, pad_signals


def _mat_field(data: dict, name: str, *alternatives: str):
    for candidate in (name, *alternatives):
        if candidate in data:
            return data[candidate]
    lowered = {str(key).lower(): key for key in data}
    for candidate in (name, *alternatives):
        key = lowered.get(candidate.lower())
        if key is not None:
            return data[key]
    raise KeyError(f"Missing NinaPro field {name!r}; available fields: {sorted(data)}")


def _find_mat(root: Path, subject: int, exercise: int) -> Path:
    filename = f"S{subject}_E{exercise}_A1.mat"
    candidates = [root / f"s{subject}" / filename, root / filename]
    candidates.extend(sorted(root.glob(f"**/{filename}")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} below {root}. "
        "Please extract the official NinaPro subject archives first."
    )


def _resample_signal(
    signal: np.ndarray,
    source_frequency: int,
    target_frequency: int,
) -> np.ndarray:
    if source_frequency == target_frequency:
        return signal.astype(np.float32, copy=False)
    ratio = Fraction(target_frequency, source_frequency).limit_denominator()
    return resample_poly(
        signal,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
    ).astype(np.float32)


def _extract_segment(
    emg: np.ndarray,
    labels: np.ndarray,
    repetitions: np.ndarray,
    gesture_id: int,
    repetition_id: int,
) -> np.ndarray | None:
    runs = contiguous_runs(
        (labels == gesture_id) & (repetitions == repetition_id)
    )
    if not runs:
        return None
    start, end = max(runs, key=lambda item: item[1] - item[0])
    return emg[start:end]


def load_ninapro_subject(
    root: Path,
    subject: int,
    exercise: int = 1,
    gesture_ids=range(1, 11),
    target_frequency: int = 200,
    label_field: str = "restimulus",
    repetition_field: str = "rerepetition",
) -> CachedSubject:
    """Load the same ten gesture classes from a DB2, DB3, or DB4 subject."""
    signals: list[np.ndarray] = []
    labels_out: list[int] = []
    repetitions_out: list[int] = []
    rest_segments: list[np.ndarray] = []

    for current_exercise in [exercise]:
        mat_path = _find_mat(root, subject, int(current_exercise))
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        emg = np.asarray(_mat_field(data, "emg", "EMG"), dtype=np.float32)
        if emg.ndim != 2:
            raise ValueError(f"Expected a 2D EMG array in {mat_path}, got {emg.shape}")
        if emg.shape[1] != 12 and emg.shape[0] == 12:
            emg = emg.T
        if emg.shape[1] != 12:
            raise ValueError(f"Expected 12 EMG channels in {mat_path}, got {emg.shape}")

        labels = np.asarray(_mat_field(data, label_field, "restimulus")).reshape(-1)
        repetitions = np.asarray(
            _mat_field(data, repetition_field, "rerepetition")
        ).reshape(-1)
        if len(labels) != len(emg) or len(repetitions) != len(emg):
            raise ValueError(
                f"EMG/label length mismatch in {mat_path}: "
                f"{emg.shape}, {labels.shape}, {repetitions.shape}"
            )

        try:
            source_frequency = int(
                np.asarray(_mat_field(data, "frequency", "sampling_frequency")).item()
            )
        except (KeyError, ValueError):
            source_frequency = 2000

        for gesture_id in gesture_ids:
            for repetition_id in sorted(
                int(value)
                for value in np.unique(repetitions[labels == gesture_id])
                if int(value) > 0
            ):
                segment = _extract_segment(
                    emg, labels, repetitions, int(gesture_id), repetition_id
                )
                if segment is None or len(segment) == 0:
                    continue
                signals.append(
                    _resample_signal(segment, source_frequency, target_frequency)
                )
                labels_out.append(int(gesture_id) - 1)
                repetitions_out.append(repetition_id)

        for start, end in contiguous_runs(labels == 0):
            if end - start >= 50:
                rest_segments.append(
                    _resample_signal(emg[start:end], source_frequency, target_frequency)
                )

    if not signals:
        raise ValueError(f"No requested gesture segments found for subject {subject} below {root}")
    if not rest_segments:
        raise ValueError(f"No usable rest segment found for subject {subject} below {root}")

    padded_signals, lengths = pad_signals(signals, channels=12)
    padded_rest, rest_lengths = pad_signals(rest_segments, channels=12)
    return CachedSubject(
        subject=subject,
        signals=padded_signals,
        lengths=lengths,
        labels=np.asarray(labels_out, dtype=np.int64),
        repetitions=np.asarray(repetitions_out, dtype=np.int16),
        rest_signals=padded_rest,
        rest_lengths=rest_lengths,
        frequency=target_frequency,
    )
