"""Value resolution and range limits of one VitalDB track, computed at import time.

For every waveform and numeric track this gives the value step (the resolution in physical units) and, for
integer waveforms, the converter range and where the signal sits at a range limit.

Integer waveforms (SNUADC, Primus) store counts: ``physical = count * gain + offset``. The step is ``gain``.
The converter's width is the smallest whole number of bits that holds the largest count seen (SNUADC reaches
count 1023: a 10-bit converter in 16-bit storage), so the converter range is
``[offset, offset + gain * (2**bits - 1)]``. A range limit is a count at which the signal stays for at least
``min_run_s``: the converter's own limits (count 0 or 2**bits - 1), or the highest or lowest count seen when
the signal plateaus there (an upstream output stage that saturates first, e.g. the arterial pressure output
near 400 mmHg). Not every limit is clipping: a capnogram sits at count 0 during every inspiration.

Float tracks (BIS EEG, every numeric) store physical values. The step is the lattice step when the values sit
exactly on one (:mod:`.lattice`), otherwise the smallest step between distinct values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from vitaldb_reader import lattice as lattice_mod
from vitaldb_reader.vital_file import FORMAT_DTYPES, TRACK_WAVE, VitalTrack


#: A plateau of at least this long (25 samples at 500 Hz) at a limit counts as a run.
MIN_RUN_S = 0.05
#: Limit runs closer than this are merged into one interval.
MERGE_GAP_S = 1.0
#: A numeric record this close to the previous one with the same value is a repeat (ST pairs: <= 20 ms).
DUPLICATE_TOLERANCE_S = 0.05
_MIN_RUN_SAMPLES = 2
_MAX_UNIQUE = 500_000
_COARSER = 1.5


@dataclass
class LimitRun:
    """One stretch where an integer waveform stays at a range limit."""

    start_s: float
    """Start, in seconds from the case origin."""
    end_s: float
    """End, in seconds from the case origin."""
    side: str
    """``"high"`` or ``"low"``."""
    count: int
    """The stored count at the limit."""
    physical: float
    """The limit in physical units."""
    kind: str
    """``"converter"`` (the converter's own limit) or ``"observed"`` (an upstream plateau)."""
    samples: int
    """Samples at the limit in the merged stretch."""


@dataclass
class TrackResolution:
    """The value resolution of one track, and for integer waveforms its range and range limits."""

    track: str
    """``device/name`` of the track."""
    kind: str
    """``"wave"`` or ``"numeric"``."""
    unit: str
    """The track's unit as the file spells it."""
    step: float | None
    """Value resolution in physical units."""
    step_source: str
    """``"gain"``, ``"lattice"``, ``"min_step"`` or ``"none"``."""
    storage: str
    """Stored dtype name."""
    gain: float
    """Scale from count to physical value."""
    offset: float
    """Physical value of count 0."""
    converter_bits: int | None = None
    """Smallest bit width that holds the largest count seen: the converter's width when the signal
    reaches its top (SNUADC: 10), otherwise only a lower bound (Primus)."""
    converter_range: list[float] | None = None
    """Physical values of count 0 and 2**bits - 1."""
    used_range: list[float] | None = None
    """Physical p0.1 and p99.9."""
    effective_bits: float | None = None
    """log2 of the number of counts between p0.1 and p99.9."""
    distinct_counts: int | None = None
    """Number of distinct counts used."""
    storage_grid: float | None = None
    """Float tracks: the lattice the values are stored on, when they sit exactly on one."""
    limits: dict = field(default_factory=dict)
    """Side -> {"count", "physical", "kind", "samples", "share"} for each limit the signal stays at."""
    runs: list[LimitRun] = field(default_factory=list)
    """Every merged stretch at a limit."""

    def summary(self) -> dict:
        """Describe the resolution in a JSON-ready form, without the individual runs.

        Returns:
            The fields, with the number of runs and their total length in seconds.
        """
        out = {k: v for k, v in self.__dict__.items() if k != "runs"}
        out["limit_runs"] = len(self.runs)
        out["limit_seconds"] = round(sum(r.end_s - r.start_s for r in self.runs), 3)
        return out


def _runs(x: np.ndarray, level: int, min_run: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` sample index pairs of runs of at least ``min_run`` samples equal to ``level``."""
    m = np.r_[False, x == level, False]
    d = np.diff(m.astype(np.int8))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = (en - st) >= min_run
    return list(zip(st[k].tolist(), en[k].tolist(), strict=True))


def _sample_times(track: VitalTrack) -> np.ndarray:
    """Return the file time of every sample of a waveform track."""
    rec = np.repeat(np.arange(track.record_times.size), track.record_sizes)
    within = np.arange(rec.size) - track.record_offsets[rec]
    return track.record_times[rec] + within / float(track.srate)


def _integer_range(res: TrackResolution, x: np.ndarray) -> None:
    """Fill the step, converter width and range, and the range used, from integer counts."""
    lo_c, hi_c = int(x.min()), int(x.max())
    bits = max(1, math.ceil(math.log2(hi_c + 1))) if hi_c > 0 else 1
    g, b = res.gain, res.offset
    res.step, res.step_source = g, "gain"
    res.converter_bits = bits
    res.converter_range = [b, b + g * (2**bits - 1)]
    bc = np.bincount(x - lo_c)
    cs = np.cumsum(bc) / x.size
    p001 = lo_c + int(np.searchsorted(cs, 0.001))
    p999 = lo_c + int(np.searchsorted(cs, 0.999))
    res.used_range = [b + g * p001, b + g * p999]
    res.effective_bits = round(math.log2(p999 - p001 + 1), 2)
    res.distinct_counts = int((bc > 0).sum())


def _limit_runs(  # noqa: PLR0913, PLR0917 - the track's context stays explicit
    res: TrackResolution, track: VitalTrack, x: np.ndarray, origin: float, min_run_s: float, merge_gap_s: float
) -> None:
    """Find the limits an integer waveform stays at, and the merged stretches it spends there."""
    bits = res.converter_bits or 1
    top = 2**bits - 1
    min_run = max(_MIN_RUN_SAMPLES, round(min_run_s * float(track.srate)))
    levels = [("low", 0), ("low", int(x.min())), ("high", top), ("high", int(x.max()))]
    t = None
    for side, level in dict.fromkeys(levels):
        rr = _runs(x, level, min_run)
        if not rr:
            continue
        kind = "converter" if level in {0, top} else "observed"
        n = int(sum(e - s for s, e in rr))
        res.limits[side] = {
            "count": level,
            "physical": res.offset + res.gain * level,
            "kind": kind,
            "samples": n,
            "share": round(n / x.size, 6),
        }
        if t is None:
            t = _sample_times(track) - origin
        merged: list[list[int]] = []
        for s, e in rr:
            if merged and t[s] - t[merged[-1][1] - 1] <= merge_gap_s:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        res.runs.extend(
            LimitRun(
                start_s=float(t[s]),
                end_s=float(t[e - 1] + 1 / float(track.srate)),
                side=side,
                count=level,
                physical=res.offset + res.gain * level,
                kind=kind,
                samples=int(e - s),
            )
            for s, e in merged
        )
    res.runs.sort(key=lambda r: r.start_s)


def _float_step(res: TrackResolution, values: np.ndarray) -> None:
    """Fill the step of a float track: its lattice step, or the smallest step between distinct values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < _MIN_RUN_SAMPLES:
        return
    d = np.diff(np.unique(v[:_MAX_UNIQUE]))
    d = d[d > 0]
    min_step = float(d.min()) if d.size else None
    found = lattice_mod.find_lattice(np.asarray(values))
    lat = float(found.step) if found is not None else None
    res.storage_grid = lat
    # the lattice is the grid values are stored on; the values may use only every k-th point of it
    # (BIS EEG: 0.05 uV steps stored on a 0.01 grid), so the resolution is the larger of the two
    if lat is not None and (min_step is None or min_step < _COARSER * lat):
        res.step, res.step_source = lat, "lattice"
    elif min_step is not None:
        res.step, res.step_source = min_step, "min_step"


def track_resolution(
    track: VitalTrack, origin: float, *, min_run_s: float = MIN_RUN_S, merge_gap_s: float = MERGE_GAP_S
) -> TrackResolution | None:
    """Compute the resolution, converter range and range-limit runs of one waveform or numeric track.

    Args:
        track: A waveform or numeric track.
        origin: The case origin; run times are seconds from it.
        min_run_s: Shortest stay at a limit that counts as a run.
        merge_gap_s: Runs closer than this are merged.

    Returns:
        The track's resolution, or ``None`` for a string track or an empty one.
    """
    if track.dtype is None or not len(track.values):
        return None
    kind = "wave" if track.track_type == TRACK_WAVE else "numeric"
    res = TrackResolution(
        track=track.dtname,
        kind=kind,
        unit=str(track.unit),
        step=None,
        step_source="none",
        storage=FORMAT_DTYPES[track.fmt].name,
        gain=float(track.gain),
        offset=float(track.offset),
    )
    values = np.asarray(track.values)
    if track.is_integer_format:
        x = values.astype(np.int64)
        _integer_range(res, x)
        if kind == "wave":
            _limit_runs(res, track, x, origin, min_run_s, merge_gap_s)
        return res
    _float_step(res, values)
    return res


def near_duplicates(times: np.ndarray, values: np.ndarray, tolerance: float) -> np.ndarray:
    """Mark records that repeat the previous record's value within ``tolerance`` of its time stamp.

    Solar8000 ST tracks write every record twice, 0-20 ms apart (median 1 ms) and with the same value in
    ~97% of pairs. A repeat with a different value is data and is kept, as is anything further apart.

    Args:
        times: Record times in file order, in any unit.
        values: Record values.
        tolerance: Largest time difference, in the unit of ``times``, that counts as a repeat.

    Returns:
        A mask, ``True`` for records to keep.
    """
    keep = np.ones(times.size, dtype=bool)
    if times.size > 1:
        close = np.abs(np.diff(times)) <= tolerance
        if values.dtype.kind in "fc":
            same_v = (values[1:] == values[:-1]) | (np.isnan(values[1:]) & np.isnan(values[:-1]))
        else:
            same_v = values[1:] == values[:-1]
        keep[1:] = ~(close & same_v)
    return keep
