from __future__ import annotations

from pathlib import Path
from fractions import Fraction

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from .data import CachedSubject, contiguous_runs, pad_signals


def resample_signal(
    signal: np.ndarray,
    source_frequency: int,
    target_frequency: int,
) -> np.ndarray:

    if source_frequency == target_frequency:
        return signal.astype(np.float32)


    ratio = Fraction(
        target_frequency,
        source_frequency,
    ).limit_denominator()


    result = resample_poly(
        signal,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
    )


    return result.astype(np.float32)



def extract_segment(
    emg: np.ndarray,
    labels: np.ndarray,
    repetitions: np.ndarray,
    gesture_id: int,
    repetition_id: int,
):
    """
    Extract one gesture repetition.
    """

    mask = (
        (labels == gesture_id)
        &
        (repetitions == repetition_id)
    )


    runs = contiguous_runs(mask)


    if len(runs) == 0:
        return None


    # DB4 occasionally has multiple runs.
    # Keep the longest one.
    if len(runs) > 1:

        lengths = [
            end-start
            for start,end in runs
        ]

        idx = int(np.argmax(lengths))

        start,end = runs[idx]

    else:

        start,end = runs[0]


    return emg[start:end]



def load_db4_subject(
    root: Path,
    subject: int,
    exercise: int = 1,
    gesture_ids=range(1,11),
    target_frequency: int = 200,
    label_field: str = "restimulus",
    repetition_field: str = "rerepetition",
) -> CachedSubject:


    mat_path = (
        root
        /
        f"s{subject}"
        /
        f"S{subject}_E{exercise}_A1.mat"
    )


    if not mat_path.exists():
        raise FileNotFoundError(
            mat_path
        )


    data = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )


    emg = np.asarray(
        data["emg"],
        dtype=np.float32,
    )


    labels = np.asarray(
        data[label_field]
    ).reshape(-1)


    repetitions = np.asarray(
        data[repetition_field]
    ).reshape(-1)


    frequency = int(
        np.asarray(
            data["frequency"]
        ).item()
    )


    if emg.shape[1] != 12:
        raise ValueError(
            f"Expected 12 channels, got {emg.shape}"
        )


    signals=[]
    labels_out=[]
    reps_out=[]


    for gesture in gesture_ids:

        for repetition in range(1,7):

            segment = extract_segment(
                emg,
                labels,
                repetitions,
                gesture,
                repetition,
            )


            if segment is None:
                continue


            segment=resample_signal(
                segment,
                frequency,
                target_frequency,
            )


            signals.append(segment)

            labels_out.append(
                gesture-1
            )

            reps_out.append(
                repetition
            )


    # rest only for threshold
    rest_segments=[]


    for start,end in contiguous_runs(
        labels==0
    ):

        rest=emg[start:end]


        if len(rest)<50:
            continue


        rest=resample_signal(
            rest,
            frequency,
            target_frequency,
        )


        rest_segments.append(rest)



    padded_signals,lengths = pad_signals(
        signals,
        channels=12,
    )


    padded_rest,rest_lengths = pad_signals(
        rest_segments,
        channels=12,
    )



    return CachedSubject(

        subject=subject,

        signals=padded_signals,

        lengths=lengths,

        labels=np.asarray(
            labels_out,
            dtype=np.int64,
        ),

        repetitions=np.asarray(
            reps_out,
            dtype=np.int16,
        ),

        rest_signals=padded_rest,

        rest_lengths=rest_lengths,

        frequency=target_frequency,
    )



if __name__ == "__main__":


    root=Path(
        "/home/xuzhenpeng/work/aemg_ninapro/dataset/ninapro_db4"
    )


    subject=load_db4_subject(
        root,
        subject=1,
    )


    print(
        "subject:",
        subject.subject
    )

    print(
        "actions:",
        len(subject.samples())
    )

    print(
        "rest:",
        len(subject.rest_segments())
    )

    print(
        "frequency:",
        subject.frequency
    )

    print(
        "shape:",
        subject.samples()[0].signal.shape
    )
