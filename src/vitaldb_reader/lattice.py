"""Recover the integers under float-valued tracks, when that can be done without changing any value.

Most float tracks in VitalDB are scaled integers written as floats: a BIS EEG sample is ``k * 0.01``, a
heart rate is ``k * 1``, a propofol volume is ``k * 0.001``. Storing ``k`` with its ``step`` and
``offset`` is smaller and lets an integer codec (AtriumDB's, or Parquet's) work. It is only acceptable if
decoding gives back every original value **bit for bit**, so :func:`find_lattice` verifies exactly that,
under the one decoding rule both importers use::

    value = dtype(float64(k) * step + offset)

and returns ``None`` when it does not hold. NaN, infinities and negative zero are not on any lattice;
a track holding them stays float.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


__all__ = ["Lattice", "decode", "find_lattice"]

_MAX_DISTINCT_PROBE = 50  # smallest gaps between distinct values tried as the step
_DIVISORS = (1, 2, 3, 4, 5, 8, 10)


@dataclass(frozen=True)
class Lattice:
    """Integers ``k`` and the scale that maps them back to the original float values."""

    k: np.ndarray
    """The integers, in the smallest signed or unsigned dtype that holds them."""
    step: float
    offset: float
    float_dtype: np.dtype
    """The dtype the decoded values are cast to (the source dtype)."""


def decode(k: np.ndarray, step: float, offset: float, float_dtype: np.dtype) -> np.ndarray:
    """Map integers back to float values with the one rule both importers use.

    Args:
        k: The integers.
        step: Value per integer step.
        offset: Value at ``k == 0``.
        float_dtype: The dtype to cast to.

    Returns:
        ``float_dtype(float64(k) * step + offset)``.
    """
    return (k.astype(np.float64) * step + offset).astype(float_dtype)


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _smallest_int_dtype(lo: int, hi: int) -> np.dtype:
    for name in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "int64"):
        info = np.iinfo(name)
        if info.min <= lo and hi <= info.max:
            return np.dtype(name)
    raise OverflowError(f"integers {lo}..{hi} do not fit int64")


def _spellings(x: float) -> set[float]:
    """The spellings of a value tried as an exact step or offset.

    Returns:
        The value itself, its float32 rounding, and its short decimal spellings.
    """
    return {x, float(np.float32(x))} | {float(f"{x:.{d}g}") for d in range(1, 10)}


def find_lattice(values: np.ndarray) -> Lattice | None:
    """Find integers ``k`` with ``decode(k, step, offset) == values`` bit for bit.

    Args:
        values: A float32 or float64 array.

    Returns:
        The lattice, or ``None`` if no step and offset reproduce every value exactly.
    """
    dtype = values.dtype
    if dtype.kind != "f" or values.size == 0:
        return None
    if not np.all(np.isfinite(values)) or np.any((values == 0) & np.signbit(values)):
        return None
    distinct = np.unique(values.astype(np.float64))
    if distinct.size == 1:
        return _verified(values, np.zeros(values.size, dtype=np.int64), 1.0, float(distinct[0]))
    tolerance = 2.5 * float(np.spacing(dtype.type(np.abs(distinct).max())))
    gaps = np.unique(np.sort(np.diff(distinct))[:_MAX_DISTINCT_PROBE])
    candidates = {g / d for g in gaps for d in _DIVISORS} | {10.0**-e for e in range(7)}
    for q in sorted(candidates, reverse=True):
        if q <= tolerance:
            continue
        idx = np.round((distinct - distinct[0]) / q)
        if np.unique(idx).size != distinct.size:
            continue  # two distinct values on one lattice point: the step is too coarse
        design = np.vstack([np.ones_like(idx), idx]).T
        (origin, fitted), *_ = np.linalg.lstsq(design, distinct, rcond=None)
        if np.abs(distinct - (origin + idx * fitted)).max() > max(tolerance, 1e-12):
            continue
        # The fitted step is near the constant the device used, not equal to it: try its nearby
        # exact spellings and keep the first pair that reproduces every value.
        for step in sorted(_spellings(float(fitted)), key=lambda s: (len(repr(s)), abs(s - fitted))):
            if step <= 0:
                continue
            shift = round(origin / step)
            for offset in sorted(
                _spellings(float(origin - shift * step)) | {0.0}, key=lambda o: (abs(o) > 0, len(repr(o)))
            ):
                k = np.round((values.astype(np.float64) - offset) / step)
                if not math.isfinite(float(np.abs(k).max())):
                    continue
                found = _verified(values, k.astype(np.int64), step, offset)
                if found is not None:
                    return found
        return None
    return None


def _verified(values: np.ndarray, k: np.ndarray, step: float, offset: float) -> Lattice | None:
    if not _bits_equal(decode(k, step, offset, values.dtype), values):
        return None
    lo, hi = int(k.min()), int(k.max())
    return Lattice(k=k.astype(_smallest_int_dtype(lo, hi)), step=step, offset=offset, float_dtype=values.dtype)
