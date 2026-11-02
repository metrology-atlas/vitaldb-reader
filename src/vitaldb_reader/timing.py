"""Cut a waveform track into regularly sampled segments, under one of three timing policies.

A ``.vital`` waveform is a run of records. Each record carries a float64 start time; its samples follow
at the nominal rate. Two records join without a break when the second starts where the first ends, to
within half a sample. A gap, an overlap or a step back in time starts a new *run*.

Inside a run the recorder stamps each record as the previous stamp plus the record's duration at the
nominal rate, so a run carries no information about the device's true sample clock: it looks perfectly
sampled. (The only deviation, about 1 ppm, is float64 rounding at the year-2100 epoch the open dataset
is shifted to, where a float resolves 2**-20 s.) The device clock does drift against the recording PC
by tens of ppm, and the recorder corrects it with a jump at the start of the next run: an overlap when
the device ran fast, a short gap when it ran slow. ``RESYNC`` spreads that jump back over the run.

``TimingMode.FILE``
    Reproduce the file's time stamps. Each run is fitted with the straight line through its first and
    last record (an exact rational period), and split wherever a record would land more than
    ``tolerance_s`` from its stored time. Every sample then sits within ``tolerance_s`` of where the
    file puts it.

``TimingMode.NOMINAL``
    Use the nominal rate, exactly, and anchor each run at its first record. Runs are never split for
    drift, so samples late in a long run can sit milliseconds away from the file's stamps. That
    distance is reported per segment.

``TimingMode.RESYNC``
    Estimate each run's true rate from the re-synchronisation jump that ends it. A run that is at least
    ``min_run_s`` long and is followed by a jump of at most ``max_jump_s`` (an implied rate error of at
    most ``max_ppm``) gets the period that makes it end exactly where the next run starts:
    ``(start of next run - start of this run) / samples in this run``. Other runs (short ones, the last
    one, runs that end in a real gap) get the track's duration-weighted mean corrected rate, or the
    nominal rate if the track has no qualifying run. Each run stays anchored at its first record. The
    estimate assumes a jump undoes only drift accumulated since the previous jump, and that the
    recording PC's own clock is steady over the run.

Times here are exact :class:`~fractions.Fraction` seconds relative to the case origin. Converting to a
storage grid (whole microseconds, whole nanoseconds, nanohertz) is the importer's job, and the
comparison suite measures what that conversion costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
import itertools

import numpy as np

from vitaldb_reader.errors import VitalValidationError
from vitaldb_reader.vital_file import TRACK_WAVE, VitalTrack


__all__ = ["ResyncLimits", "Segment", "TimingMode", "nominal_rate", "runs", "segments"]


class TimingMode(StrEnum):
    """How to place a waveform's samples in time."""

    FILE = "file"
    """Reproduce the file's record time stamps to within a tolerance."""
    NOMINAL = "nominal"
    """Exactly the nominal sample rate, anchored at each run's first record."""
    RESYNC = "resync"
    """Each run's rate estimated from the re-synchronisation jump that ends it."""


@dataclass(frozen=True)
class ResyncLimits:
    """Which run boundaries count as a re-synchronisation of the device clock."""

    min_run_s: float = 60.0
    """Shortest run whose jump is used: a short run's jump says little about a rate."""
    max_jump_s: float = 2.0
    """Largest jump, either sign, treated as a re-synchronisation rather than a gap in recording."""
    max_ppm: float = 1000.0
    """Largest implied rate error accepted; beyond it the boundary is not a clock correction."""


@dataclass(frozen=True)
class Segment:
    """A regularly sampled stretch of one track."""

    sample_start: int
    """Index of the first sample in the track's value array."""
    sample_stop: int
    """One past the last sample."""
    record_start: int
    record_stop: int
    start_s: Fraction
    """Time of the first sample, in seconds from the case origin, exact."""
    period_s: Fraction
    """Seconds between samples, exact."""
    max_error_s: float
    """Largest distance between a sample's time here and where the file puts it."""
    rate_source: str = "nominal"
    """Where the period came from: ``nominal``, ``fit`` (FILE), ``resync`` (this run's own jump) or
    ``track`` (the track's mean corrected rate)."""

    @property
    def ppm(self) -> float:
        """The segment's rate relative to nominal, in parts per million (set by :func:`segments`)."""
        return self._ppm

    _ppm: float = 0.0

    @property
    def n_samples(self) -> int:
        """Samples in this segment."""
        return self.sample_stop - self.sample_start

    @property
    def rate_hz(self) -> Fraction:
        """The segment's sample rate, exact."""
        return 1 / self.period_s


def nominal_rate(track: VitalTrack) -> Fraction:
    """The nominal rate as an exact fraction, read from the decimal spelling of the stored float32.

    500.0 reads as 500 and 62.5 as 125/2; the float32 itself is not used, because a float32 like 0.1 Hz
    is not the rate anyone meant.

    Args:
        track: A waveform track.

    Returns:
        The rate in Hz.

    Raises:
        VitalValidationError: If the track states no positive rate.
    """
    if track.srate <= 0:
        raise VitalValidationError(f"waveform track {track.dtname!r} states no sample rate")
    return Fraction(np.format_float_positional(np.float32(track.srate), trim="-"))


def runs(track: VitalTrack) -> list[tuple[int, int]]:
    """Split a waveform's records into runs that join to within half a sample.

    Args:
        track: A waveform track.

    Returns:
        ``(first_record, one_past_last_record)`` per run, in file order.
    """
    t = track.record_times
    if t.size == 0:
        return []
    period = 1.0 / float(track.srate)
    ends = t + track.record_sizes * period
    breaks = np.flatnonzero(np.abs(t[1:] - ends[:-1]) >= 0.5 * period) + 1
    edges = np.r_[0, breaks, t.size]
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def _fit(  # noqa: PLR0913, PLR0917 - a recursive helper; the fixed fit inputs travel with it
    rel: np.ndarray,
    first: np.ndarray,
    a: int,
    b: int,
    nominal: Fraction,
    tolerance_s: float,
    out: list[tuple[int, int, Fraction, float]],
) -> None:
    """Fit records ``[a, b)`` with one line, splitting recursively where it misses by more than tolerance."""
    span = int(first[b - 1] - first[a])
    if b - a == 1 or span == 0:
        out.append((a, b, nominal, 0.0))
        return
    period = (Fraction(float(rel[b - 1])) - Fraction(float(rel[a]))) / span
    p = float(period)
    predicted = rel[a] + (first[a:b] - first[a]) * p
    miss = np.abs(rel[a:b] - predicted)
    worst = int(np.argmax(miss))
    if miss[worst] <= tolerance_s:
        out.append((a, b, period, float(miss[worst])))
        return
    cut = a + max(worst, 1)
    _fit(rel, first, a, cut, nominal, tolerance_s, out)
    _fit(rel, first, cut, b, nominal, tolerance_s, out)


def _resync_periods(  # noqa: PLR0913, PLR0917 - the run table travels together
    rel: np.ndarray,
    first: np.ndarray,
    track: VitalTrack,
    run_list: list[tuple[int, int]],
    nominal: Fraction,
    limits: ResyncLimits,
) -> list[tuple[Fraction, str]]:
    """Give each run a period, from the re-synchronisation jump at the start of the following run.

    Returns:
        One (period, source) per run, source being resync, track or nominal.
    """
    nominal_s = float(nominal)
    own: list[Fraction | None] = []
    weighted, weight = Fraction(0), Fraction(0)
    for i, (a, b) in enumerate(run_list):
        n = int(track.record_offsets[b] - first[a])
        period: Fraction | None = None
        if i + 1 < len(run_list) and n > 0 and n * nominal_s >= limits.min_run_s:
            nxt = run_list[i + 1][0]
            jump = float(rel[nxt]) - (float(rel[a]) + n * nominal_s)
            if abs(jump) <= limits.max_jump_s and abs(jump) / (n * nominal_s) <= limits.max_ppm * 1e-6:
                period = (Fraction(float(rel[nxt])) - Fraction(float(rel[a]))) / n
                weighted += period * n
                weight += n
        own.append(period)
    fallback, source = (weighted / weight, "track") if weight else (nominal, "nominal")
    return [(p, "resync") if p is not None else (fallback, source) for p in own]


def segments(
    track: VitalTrack,
    origin: float,
    mode: TimingMode,
    *,
    tolerance_s: float = 1e-6,
    limits: ResyncLimits | None = None,
) -> list[Segment]:
    """Cut a waveform track into regularly sampled segments.

    Args:
        track: A waveform track.
        origin: The case's relative zero, in the file's seconds.
        mode: The timing policy.
        tolerance_s: In ``FILE`` mode, the largest distance allowed between a record's stored time and
            its time on the fitted line. Ignored in the other modes.
        limits: In ``RESYNC`` mode, which boundaries count as re-synchronisations.

    Returns:
        Segments in file order.

    Raises:
        VitalValidationError: If the track is not a waveform.
    """
    if track.track_type != TRACK_WAVE:
        raise VitalValidationError(f"{track.dtname!r} is not a waveform track")
    rate = nominal_rate(track)
    nominal = 1 / rate
    rel = track.record_times - origin  # exact: both are doubles of the same magnitude
    first = track.record_offsets[:-1]
    result: list[Segment] = []
    run_list = runs(track)
    resync = (
        _resync_periods(rel, first, track, run_list, nominal, limits or ResyncLimits())
        if mode is TimingMode.RESYNC
        else None
    )
    for i, (a, b) in enumerate(run_list):
        pieces: list[tuple[int, int, Fraction, float]] = []
        source = "nominal"
        if mode is TimingMode.FILE:
            _fit(rel, first, a, b, nominal, tolerance_s, pieces)
            source = "fit"
        else:
            period_run, source = resync[i] if resync is not None else (nominal, "nominal")
            predicted = rel[a] + (first[a:b] - first[a]) * float(period_run)
            pieces.append((a, b, period_run, float(np.abs(rel[a:b] - predicted).max())))
        for ra, rb, period, miss in pieces:
            if track.record_offsets[rb] == first[ra]:
                continue  # records with no samples place nothing
            # within a record, sample j sits at t + j/srate in the file and at t + j*period here
            tail = float(abs(period - nominal)) * float(track.record_sizes[ra:rb].max() - 1)
            result.append(
                Segment(
                    sample_start=int(first[ra]),
                    sample_stop=int(track.record_offsets[rb]),
                    record_start=ra,
                    record_stop=rb,
                    start_s=Fraction(float(rel[ra])),
                    period_s=period,
                    max_error_s=miss + tail,
                    rate_source=source,
                    _ppm=float((nominal / period - 1) * 1_000_000),
                )
            )
    return result
