from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import yaml
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Sample:
    """
    One EMG gesture sample.

    signal:
        [time, channels]

    label:
        gesture class id

    subject:
        subject/user id

    repetition:
        repetition index
    """

    signal: np.ndarray
    label: int
    subject: int
    repetition: int


@dataclass(slots=True)
class CachedSubject:
    """
    Unified representation for all EMG datasets.

    Dataset adapters (DB4/EPN612)
    must convert their raw data into this format.
    """

    subject: int

    # padded action signals
    # [N, max_length, channels]
    signals: np.ndarray

    # original lengths
    lengths: np.ndarray

    labels: np.ndarray

    repetitions: np.ndarray


    # rest calibration signals
    rest_signals: np.ndarray

    rest_lengths: np.ndarray


    frequency: int


    def samples(self) -> list[Sample]:

        result = []

        for i, length in enumerate(self.lengths):

            signal = self.signals[
                i,
                :int(length)
            ].copy()


            result.append(
                Sample(
                    signal=signal,
                    label=int(self.labels[i]),
                    subject=self.subject,
                    repetition=int(
                        self.repetitions[i]
                    ),
                )
            )


        return result



    def rest_segments(self) -> list[np.ndarray]:

        result = []

        for i, length in enumerate(
            self.rest_lengths
        ):

            result.append(
                self.rest_signals[
                    i,
                    :int(length)
                ].copy()
            )

        return result



def contiguous_runs(
    mask: np.ndarray,
) -> list[tuple[int,int]]:
    """
    Find continuous True regions.

    Example:
        [0,1,1,1,0,1]

    return:
        [(1,4),(5,6)]
    """

    indices = np.flatnonzero(mask)


    if len(indices)==0:
        return []


    split_points = (
        np.flatnonzero(
            np.diff(indices)>1
        )
        + 1
    )


    groups=np.split(
        indices,
        split_points
    )


    return [
        (
            int(g[0]),
            int(g[-1])+1
        )
        for g in groups
    ]



def pad_signals(
    signals: Sequence[np.ndarray],
    channels:int
):

    if len(signals)==0:

        return (
            np.zeros(
                (0,0,channels),
                dtype=np.float32
            ),
            np.zeros(
                (0,),
                dtype=np.int32
            )
        )


    lengths=np.asarray(
        [
            s.shape[0]
            for s in signals
        ],
        dtype=np.int32
    )


    max_length=int(
        lengths.max()
    )


    padded=np.zeros(
        (
            len(signals),
            max_length,
            channels
        ),
        dtype=np.float32
    )


    for i,s in enumerate(signals):

        padded[
            i,
            :s.shape[0]
        ]=s

    return padded,lengths

def load_config(path):
    path = Path(path)

    with open(path, "r") as f:
        return yaml.safe_load(f)
