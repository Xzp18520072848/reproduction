from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CachedSubject


@dataclass(slots=True)
class NCTSentence:
    windows: np.ndarray
    attention_mask: np.ndarray
    time_positions: np.ndarray
    energies: np.ndarray


@dataclass(slots=True)
class TokenizedSubject:
    subject: int
    windows: np.ndarray
    attention_masks: np.ndarray
    time_positions: np.ndarray
    energies: np.ndarray
    labels: np.ndarray
    repetitions: np.ndarray
    threshold: float
    frequency: int
    window_samples: int
    step_samples: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            subject=np.asarray(self.subject, dtype=np.int32),
            windows=self.windows,
            attention_masks=self.attention_masks,
            time_positions=self.time_positions,
            energies=self.energies,
            labels=self.labels,
            repetitions=self.repetitions,
            threshold=np.asarray(self.threshold, dtype=np.float32),
            frequency=np.asarray(self.frequency, dtype=np.int32),
            window_samples=np.asarray(self.window_samples, dtype=np.int32),
            step_samples=np.asarray(self.step_samples, dtype=np.int32),
        )


def milliseconds_to_samples(milliseconds: int, frequency: int) -> int:
    samples = round(milliseconds * frequency / 1000)
    if samples <= 0:
        raise ValueError("Window or step length must be positive.")
    return samples


def sliding_windows(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected signal [time, channels], got {signal.shape}.")

    time_length, channels = signal.shape
    if time_length < window_samples:
        return (
            np.zeros((0, channels, window_samples), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    starts = np.arange(
        0,
        time_length - window_samples + 1,
        step_samples,
        dtype=np.int64,
    )
    windows = np.stack(
        [signal[start : start + window_samples].T for start in starts],
        axis=0,
    )
    return windows, starts


def window_energies(windows: np.ndarray) -> np.ndarray:
    """Paper Eq. (1): divide summed channel energy by window length."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows [N, C, L], got {windows.shape}.")
    if windows.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    energies = np.square(windows, dtype=np.float64).sum(axis=(1, 2))
    energies /= windows.shape[-1]
    return energies.astype(np.float32)


def normalize_windows(
    windows: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Paper Eq. (3): z-score every channel inside every token."""
    if windows.shape[0] == 0:
        return windows.astype(np.float32, copy=False)
    means = windows.mean(axis=-1, keepdims=True)
    stds = windows.std(axis=-1, keepdims=True)
    return ((windows - means) / (stds + epsilon)).astype(np.float32)


def estimate_threshold(
    rest_segments: list[np.ndarray],
    frequency: int,
    window_ms: int,
    step_ms: int,
    method: str = "percentile",
    percentile: float = 95.0,
    std_factor: float = 3.0,
) -> tuple[float, np.ndarray]:
    """Estimate the undisclosed subject calibration threshold reproducibly."""
    if not rest_segments:
        raise ValueError("No rest segment is available.")

    window_samples = milliseconds_to_samples(window_ms, frequency)
    step_samples = milliseconds_to_samples(step_ms, frequency)
    # Algorithm 1 calibrates the energy gate from the subject's resting-state
    # noise level.  A subject can contribute multiple rest trials; using only
    # the first trial makes the calibration depend on file ordering and drops
    # valid calibration evidence.
    energy_blocks = []
    for segment in rest_segments:
        windows, _ = sliding_windows(
            segment,
            window_samples=window_samples,
            step_samples=step_samples,
        )
        energies = window_energies(windows)
        if energies.size:
            energy_blocks.append(energies)
    rest_energies = (
        np.concatenate(energy_blocks, axis=0)
        if energy_blocks
        else np.zeros((0,), dtype=np.float32)
    )
    if rest_energies.size == 0:
        raise ValueError("Calibration rest is shorter than one window.")

    if method == "percentile":
        threshold = float(np.percentile(rest_energies, percentile))
    elif method == "mean_plus_std":
        threshold = float(
            rest_energies.mean() + std_factor * rest_energies.std()
        )
    else:
        raise ValueError(f"Unsupported threshold method: {method}")
    return threshold, rest_energies


def tokenize_signal(
    signal: np.ndarray,
    threshold: float,
    frequency: int,
    window_ms: int,
    step_ms: int,
    max_sequence_length: int,
    epsilon: float = 1e-6,
) -> NCTSentence:
    window_samples = milliseconds_to_samples(window_ms, frequency)
    step_samples = milliseconds_to_samples(step_ms, frequency)
    windows, starts = sliding_windows(signal, window_samples, step_samples)
    energies = window_energies(windows)
    valid = energies > threshold

    windows = normalize_windows(windows[valid], epsilon)[:max_sequence_length]
    starts = starts[valid][:max_sequence_length]
    energies = energies[valid][:max_sequence_length]
    valid_count = len(windows)
    channels = signal.shape[1]

    padded_windows = np.zeros(
        (max_sequence_length, channels, window_samples), dtype=np.float32
    )
    attention_mask = np.zeros(max_sequence_length, dtype=np.bool_)
    time_positions = np.zeros(max_sequence_length, dtype=np.float32)
    padded_energies = np.zeros(max_sequence_length, dtype=np.float32)

    if valid_count:
        padded_windows[:valid_count] = windows
        attention_mask[:valid_count] = True
        padded_energies[:valid_count] = energies
        maximum_start = max(signal.shape[0] - window_samples, 1)
        time_positions[:valid_count] = starts / maximum_start

    return NCTSentence(
        windows=padded_windows,
        attention_mask=attention_mask,
        time_positions=time_positions,
        energies=padded_energies,
    )


def tokenize_subject(
    subject_data: CachedSubject,
    window_ms: int,
    step_ms: int,
    max_sequence_length: int,
    threshold_method: str = "percentile",
    threshold_percentile: float = 95.0,
    std_factor: float = 3.0,
    threshold_override: float | None = None,
    keep_empty: bool = False,
) -> TokenizedSubject:
    estimated, rest_energies = estimate_threshold(
        subject_data.rest_segments(),
        subject_data.frequency,
        window_ms,
        step_ms,
        method=threshold_method,
        percentile=threshold_percentile,
        std_factor=std_factor,
    )
    threshold = estimated if threshold_override is None else threshold_override

    sentences: list[NCTSentence] = []
    valid_samples = []
    for sample in subject_data.samples():
        sentence = tokenize_signal(
            sample.signal,
            threshold,
            subject_data.frequency,
            window_ms,
            step_ms,
            max_sequence_length,
        )
        if not sentence.attention_mask.any():
            if not keep_empty:
                continue
            # Preserve the labelled recording for a downstream adapter, but do
            # not invent a contraction token.  Algorithm 1 pads a short NCT
            # sentence with zero tokens; it never turns padding into a valid
            # contraction when the energy gate rejected every window.
        sentences.append(sentence)
        valid_samples.append(sample)

    if not sentences:
        raise ValueError(f"Subject {subject_data.subject} has no NCT sentences.")

    counts = np.asarray(
        [sentence.attention_mask.sum() for sentence in sentences]
    )
    print(
        f"S{subject_data.subject}: threshold={threshold:.6g}, "
        f"rest_mean={rest_energies.mean():.6g}, "
        f"tokens={counts.min()}/{counts.mean():.1f}/{counts.max()}, "
        f"kept={len(sentences)}"
    )

    return TokenizedSubject(
        subject=subject_data.subject,
        windows=np.stack([sentence.windows for sentence in sentences]),
        attention_masks=np.stack(
            [sentence.attention_mask for sentence in sentences]
        ),
        time_positions=np.stack(
            [sentence.time_positions for sentence in sentences]
        ),
        energies=np.stack([sentence.energies for sentence in sentences]),
        labels=np.asarray([sample.label for sample in valid_samples], np.int64),
        repetitions=np.asarray(
            [sample.repetition for sample in valid_samples], np.int16
        ),
        threshold=float(threshold),
        frequency=subject_data.frequency,
        window_samples=milliseconds_to_samples(window_ms, subject_data.frequency),
        step_samples=milliseconds_to_samples(step_ms, subject_data.frequency),
    )


def load_tokenized_subject(path: Path) -> TokenizedSubject:
    with np.load(path, allow_pickle=False) as data:
        return TokenizedSubject(
            subject=int(data["subject"]),
            windows=np.asarray(data["windows"], dtype=np.float32),
            attention_masks=np.asarray(data["attention_masks"], dtype=np.bool_),
            time_positions=np.asarray(data["time_positions"], dtype=np.float32),
            energies=np.asarray(data["energies"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int64),
            repetitions=np.asarray(data["repetitions"], dtype=np.int16),
            threshold=float(data["threshold"]),
            frequency=int(data["frequency"]),
            window_samples=int(data["window_samples"]),
            step_samples=int(data["step_samples"]),
        )
