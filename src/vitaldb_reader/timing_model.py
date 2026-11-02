"""Measured timing of VitalDB monitor numerics and clinical-table times, for import-time annotations.

Two things a time stamp in VitalDB does not say on its own:

Time anchors (the availability rule)
    Every value describes an interval of time and cannot exist until that interval has closed. Three
    instants describe it: the window start, the window end, and the stamp. A value is available, at the
    earliest, at the window end, and a stamp must never fall before it. This module anchors every timing
    quantity at the window end: a numeric's delay is measured from its window's end to its stamp, and its
    window length is given separately. Where a window's middle is useful (comparing two signals after the
    fact), it is a derived, descriptive quantity (``support_centre_s``), never a time stamp: re-stamping a
    value to it would place the value before it could have been known.

Numeric lag
    A monitor numeric (heart rate, arterial pressure, ...) summarises a window of the waveform and is stamped
    when the recorder receives it, after the window has ended. :data:`NUMERIC_LAG` holds, per track, the
    algorithm that best reproduces the values from the waveforms, the window's length, and the delay from the
    window's end to the stamp across cases. Delays are stable within a case but differ between cases (40-70%
    of the variance is between cases), so the table gives the spread across cases, not one exact value. Heart
    rate falls into two groups about one 2 s record apart. The BIS index is the slowest: a trailing mean of
    its own EEG over 15 s or a longer per-case setting, with further slow weight, so it suits trends over
    minutes rather than event timing. The stamps are left as they are: they are when the value became
    available. A delay below zero would mean a value stamped before its window ended, which cannot happen,
    so it points to an error in the window's end or in the clock mapping; such entries are marked
    ``provisional``.

Clinical-table times
    ``anestart``, ``aneend``, ``opstart`` and ``opend`` are anaesthesia-record (EMR) clock times entered
    either to the minute or on a five-minute clock mark. Five-minute entries are rounded up: the true time lies
    in the five minutes before the recorded one. :func:`clinical_time_resolution` places each time on its
    grid and returns the interval it stands for and the earliest time the event could have been known (the
    interval's end; documentation lag comes on top). Which grid a time is on needs the case's clock minute, which
    comes from ``adm`` (the admission date, relative to the case start truncated to the minute) and the
    seconds phase the EMR times share.

The numbers come from analyses in the vitaldb-bench repository (``tools/numeric_derivation_pilot.py``,
``tools/numeric_delay_variability.py``, ``tools/spo2_event_pilot.py``, ``tools/vigilance_window.py``,
``tools/bis_index_delay.py`` and ``tools/emr_markers.py``). :data:`MODEL_VERSION` changes whenever they do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any


#: Version of the timing model. Annotations carry it, so a later model can replace them.
MODEL_VERSION = "vitaldb-timing-2026.11.1"

_FIVE_MIN_S = 300.0
_MINUTE_S = 60.0
_DAY_S = 86400
_EMR_FIELDS = ("anestart", "aneend", "opstart", "opend")
_PHASE_QUORUM = 3
_RECORDING_FIELDS = ("casestart", "caseend")
_ROUNDING_MEASURED = ("anestart", "aneend")
# A time this close to a whole minute of the case clock (in minutes) is on the minute grid.
_GRID_TOLERANCE = 1e-3


@dataclass(frozen=True)
class NumericLag:
    """How one monitor numeric lags the physiology it summarises."""

    track: str
    """The VitalDB track, ``device/name``."""
    algorithm: str
    """The averaging that best reproduces the values from the waveform."""
    delay_from_end_s: tuple[float | None, float, float | None]
    """Delay from the window's end to the stamp across cases: 10th, 50th and 90th percentiles (s)."""
    window_s: float
    """Length of the window the value summarises (s); 0 for a single instant (a breath's end or peak). For an
    exponentially weighted mean, twice the mean age of its inputs (its effective length)."""
    cases: int
    """Cases the delay was measured in."""
    basis: str
    """``"measured"`` (fixed algorithm, many cases) or ``"pilot"`` (a smaller sample)."""
    between_case_sd_s: float | None = None
    """Spread of the delay between cases (random-effects SD, s)."""
    within_case_sd_s: float | None = None
    """Spread of the delay between 30-minute blocks of one case, including estimation noise (s)."""
    modes: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    """Groups of cases as (median delay from the window's end in s, share of cases), when the delay is
    bimodal."""
    note: str = ""
    """What a user should know beyond the numbers."""
    provisional: str = ""
    """Why the entry is not yet trustworthy, or empty. Set when a delay below zero (impossible under the
    availability rule) shows an error in the window's end or in the clock mapping."""

    @property
    def support_centre_s(self) -> float:
        """How far the middle of the typical window lies before the stamp (s).

        Descriptive only, for comparing signals after the fact. It is not a time stamp: a value moved to it
        would sit before the time it could have been known.

        Returns:
            The median delay from the window's end plus half the window.
        """
        return round(self.delay_from_end_s[1] + self.window_s / 2, 2)

    def metadata(self) -> dict[str, Any]:
        """The lag as JSON-ready annotation metadata.

        Returns:
            The fields, with the model version and how to read them.
        """
        return {
            "model_version": MODEL_VERSION,
            "track": self.track,
            "algorithm": self.algorithm,
            "delay_from_window_end_s_p10_p50_p90": list(self.delay_from_end_s),
            "window_s": self.window_s,
            "support_centre_s": self.support_centre_s,
            "between_case_sd_s": self.between_case_sd_s,
            "within_case_sd_s": self.within_case_sd_s,
            "modes": [{"delay_from_window_end_s": d, "share": s} for d, s in self.modes],
            "cases": self.cases,
            "basis": self.basis,
            "provisional": self.provisional,
            "note": self.note,
            "reading": (
                "Availability rule: a value cannot be known before the end of the window it summarises. The "
                "stamp is when the value became available and is not moved. The value summarises a window "
                "window_s long that ended delay_from_window_end_s before the stamp; the case's own delay can "
                "differ by the between-case spread. support_centre_s is descriptive only: never re-stamp a "
                "value to it, which would place it before it could have been known."
            ),
        }


# Window lengths. For an EWMA with weight a per beat the mean age of the data is (1 - a) / a beats, taken at
# 75 beats per minute (0.8 s per beat); its effective window is twice that.
_BEAT_S = 0.8

NUMERIC_LAG: dict[str, NumericLag] = {
    lag.track: lag
    for lag in (
        NumericLag(
            track="Solar8000/HR",
            algorithm="60 / mean of the last 8 R-R intervals within 15% of their median, truncated to 1 bpm",
            delay_from_end_s=(1.6, 3.85, 4.65),
            window_s=round(8 * _BEAT_S, 1),
            cases=6233,
            basis="measured",
            between_case_sd_s=1.23,
            within_case_sd_s=0.78,
            modes=((1.88, 0.39), (4.17, 0.61)),
            note=(
                "Two groups about one 2 s Solar8000 record apart, fixed within a case; not the averaging window. "
                "The group follows the recording PC (ports of the Solar8000, SNUADC, Primus and BIS): the PC's "
                "majority group is right for 94% of cases. Any PC can give a late case, only some give early ones. "
                "The case's own group is the hr_delay_group label."
            ),
        ),
        NumericLag(
            track="Solar8000/ART_SBP",
            algorithm="median of per-beat maxima over the last 3 s",
            delay_from_end_s=(0.3, 1.0, 1.9),
            window_s=3.0,
            cases=493,
            basis="measured",
            between_case_sd_s=0.82,
            within_case_sd_s=0.65,
            note="Shares a per-case offset with ART_MBP and ART_DBP (rank correlation 0.86-0.92).",
        ),
        NumericLag(
            track="Solar8000/ART_MBP",
            algorithm="exponentially weighted mean of per-beat means, weight 1/4 per beat",
            delay_from_end_s=(1.0, 1.7, 2.5),
            window_s=round(2 * 3 * _BEAT_S, 1),
            cases=493,
            basis="measured",
            between_case_sd_s=0.78,
            within_case_sd_s=0.62,
            note="Effective window approximate (twice the mean data age at 75 bpm).",
        ),
        NumericLag(
            track="Solar8000/ART_DBP",
            algorithm="exponentially weighted mean of per-beat minima, weight 1/2 per beat",
            delay_from_end_s=(0.3, 1.2, 2.0),
            window_s=round(2 * 1 * _BEAT_S, 1),
            cases=495,
            basis="measured",
            between_case_sd_s=0.78,
            within_case_sd_s=0.60,
            note="Effective window approximate (twice the mean data age at 75 bpm).",
        ),
        NumericLag(
            track="Solar8000/CVP",
            algorithm="mean of the CVP waveform over the last 4 s",
            delay_from_end_s=(3.1, 3.8, 4.7),
            window_s=4.0,
            cases=177,
            basis="measured",
            between_case_sd_s=0.59,
            within_case_sd_s=0.72,
        ),
        NumericLag(
            track="Primus/ETCO2",
            algorithm="end-tidal value of the last breath",
            delay_from_end_s=(0.75, 1.55, 3.4),
            window_s=0.0,
            cases=6186,
            basis="measured",
            between_case_sd_s=0.94,
            within_case_sd_s=0.80,
            note=(
                "Measured from the end of the breath's expiratory plateau (CO2 falling through half its range) on "
                "the Primus CO2 waveform, mapped to the recorder clock with an assumed 115 ppm drift. The Primus is "
                "polled about every 7 s and reports the last completed breath; an earlier fit that also let the "
                "value's computation float over that 7 s gave delays below zero (-1.5 s median), an impossibility "
                "corrected on 13 October."
            ),
        ),
        NumericLag(
            track="Primus/PIP_MBAR",
            algorithm="peak airway pressure of the last breath",
            delay_from_end_s=(3.65, 4.6, 6.5),
            window_s=0.0,
            cases=5894,
            basis="measured",
            between_case_sd_s=0.96,
            within_case_sd_s=0.86,
            note=(
                "Measured from the end of inspiration (pressure falling through half way from peak to PEEP), when "
                "the breath's peak is known, on the Primus pressure waveform (assumed 115 ppm drift, as ETCO2). "
                "The value of the last completed breath, reported at each ~7 s poll."
            ),
        ),
        NumericLag(
            track="Primus/RR_CO2",
            algorithm="60 / mean breath interval over the last 30 s, rounded",
            delay_from_end_s=(1.6, 4.3, 7.6),
            window_s=30.0,
            cases=862,
            basis="measured",
            between_case_sd_s=1.97,
            within_case_sd_s=1.39,
        ),
        NumericLag(
            track="BIS/BIS",
            algorithm=(
                "trailing mean of the monitor's EEG-derived index over 15 s or a longer per-case setting "
                "(not recorded), plus slower components"
            ),
            delay_from_end_s=(3.0, 3.0, 4.0),
            window_s=19.0,
            cases=3297,
            basis="measured",
            between_case_sd_s=8.81,
            within_case_sd_s=6.85,
            note=(
                "Measured against the BIS monitor's own EEG (BIS/EEG1_WAV), so the lag is on the BIS clock; "
                "apply the BIS-SNUADC offset to compare with other devices. The window is the fast part's median "
                "(EEG covered by a 15 s or longer trailing mean, 2 s EEG windows included). Two settings: a 15 s "
                "trailing mean (48% of well-fitted cases, window about 16 s) and a longer one (40%, window 32-61 "
                "s, median 50); five recording PCs always use the longer one. Fits whose fast part would start "
                "before the EEG window ends (impossible) are excluded; most were longer-setting fits, so that "
                "group is the less certain. Half the index's weight is older than 30 s (10th-90th percentile "
                "17-49 s). The case's own setting is the bis_smoothing label. Use the "
                "index for trends over minutes; time events from the EEG."
            ),
        ),
        NumericLag(
            track="Solar8000/PLETH_HR",
            algorithm="pulse rate over about the last 12 beats, rounded",
            delay_from_end_s=(None, 3.4, None),
            window_s=round(12 * _BEAT_S, 1),
            cases=231,
            basis="pilot",
            note="From step events (probe off/on, artefact); the window is approximate.",
        ),
        NumericLag(
            track="Vigilance/HR_AVG",
            algorithm="running median of heart rate over about 10-36 minutes",
            delay_from_end_s=(0.0, 51.4, 174.5),
            window_s=1050.0,
            cases=52,
            basis="pilot",
            note=(
                "Describes roughly the last 10-36 minutes (window 10th-90th percentile 600-2185 s); do not align "
                "it to better than minutes. Refitted 13 October with the delay held at zero or above and a finer "
                "window grid: the earlier fit traded a longer window for a negative delay."
            ),
            provisional=(
                "Pilot of 52 cases, and in 16 of them the delay sits at zero: the data would place the value "
                "before its window ended, an unidentified offset (perhaps a heart-rate source other than the "
                "SNUADC ECG)."
            ),
        ),
    )
}


def numeric_lag(track: str) -> NumericLag | None:
    """The measured lag of a numeric track.

    Args:
        track: The VitalDB track name, ``device/name``.

    Returns:
        The lag, or ``None`` when the track has not been measured.
    """
    return NUMERIC_LAG.get(track)


def _float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def case_clock_minute(clinical: dict[str, str]) -> float | None:
    """Seconds after local midnight of the case start on the EMR clock, from the clinical row.

    ``adm`` is the admission date (local midnight) relative to the case start truncated to the minute, so
    ``-adm mod 86400`` is the case start's clock minute. The seconds come from the phase that at least three
    of the EMR times share. Only whether an EMR time sits on a five-minute mark is used from it.

    Args:
        clinical: One row of ``clinical_data.csv``.

    Returns:
        The case start's time of day in seconds, or ``None`` when it cannot be recovered.
    """
    adm = _float(clinical.get("adm"))
    if adm is None or round(adm) % 60:
        return None
    times = [_float(clinical.get(k)) for k in _EMR_FIELDS]
    phases = Counter(round(t) % 60 for t in times if t is not None)
    if not phases:
        return None
    phase, n = phases.most_common(1)[0]
    if n < _PHASE_QUORUM:
        return None
    return float((round(-adm) % _DAY_S + (-phase) % 60) % _DAY_S)


def clinical_time_resolution(clinical: dict[str, str], key: str) -> dict[str, Any] | None:
    """The interval a clinical-table time stands for.

    Args:
        clinical: One row of ``clinical_data.csv``.
        key: ``anestart``, ``aneend``, ``opstart``, ``opend``, ``casestart`` or ``caseend``.

    Returns:
        ``grid`` (``"5 min"``, ``"minute"``, ``"recording"`` or ``"unknown"``), ``rounding``, ``basis``,
        the recorded time and the interval the event lies in (``interval_s``, seconds from the case start),
        ``available_s``, the earliest time the event could have been known (the interval's end; any
        documentation lag comes on top), an estimate of the event's own time with its standard deviation
        (``event_estimate_s``, ``sd_s``: descriptive, not a time stamp), and the model version; ``None`` when
        the field is empty.
    """
    t = _float(clinical.get(key))
    if t is None:
        return None
    out: dict[str, Any] = {"model_version": MODEL_VERSION, "field": key, "recorded_s": t}
    if key in _RECORDING_FIELDS:
        res = {**out, **_recording_time(key, t)}
    elif key in _EMR_FIELDS:
        res = {**out, **_emr_time(key, t, case_clock_minute(clinical))}
    else:
        return None
    res["available_s"] = max(t, res["interval_s"][1])
    return res


def _recording_time(key: str, t: float) -> dict[str, Any]:
    if key == "casestart":
        return {
            "grid": "recording",
            "rounding": "exact",
            "basis": "definition",
            "interval_s": [t, t],
            "event_estimate_s": t,
            "sd_s": 0.0,
        }
    return {
        "grid": "recording",
        "rounding": "nearest second",
        "basis": "measured",
        "interval_s": [t - 0.5, t + 0.5],
        "event_estimate_s": t,
        "sd_s": round(1 / math.sqrt(12), 3),
    }


def _emr_time(key: str, t: float, tod: float | None) -> dict[str, Any]:
    unknown = {
        "grid": "unknown",
        "rounding": "unknown",
        "interval_s": [t - _FIVE_MIN_S, t + _MINUTE_S],
        "event_estimate_s": None,
        "sd_s": None,
    }
    if tod is None:
        return {
            **unknown,
            "basis": "no case clock",
            "note": "Grid unknown: covers a five-minute entry rounded up and a minute entry either way.",
        }
    minute = (tod + t) / _MINUTE_S
    if abs(minute - round(minute)) > _GRID_TOLERANCE:
        return {**unknown, "basis": "off the minute grid", "note": "Not on the case's minute grid."}
    if round(minute) % 5 == 0:
        return {
            "grid": "5 min",
            "rounding": "up",
            "basis": "measured" if key in _ROUNDING_MEASURED else "assumed (as anestart and aneend)",
            "interval_s": [t - _FIVE_MIN_S, t],
            "event_estimate_s": t - _FIVE_MIN_S / 2,
            "sd_s": round(_FIVE_MIN_S / math.sqrt(12), 1),
            "note": "Entered on a five-minute clock mark, rounded up. Documentation lag is not included.",
        }
    return {
        "grid": "minute",
        "rounding": "unknown",
        "basis": "rounding direction not measured",
        "interval_s": [t - _MINUTE_S, t + _MINUTE_S],
        "event_estimate_s": t,
        "sd_s": round(2 * _MINUTE_S / math.sqrt(12), 1),
        "note": "Entered to the minute. Documentation lag is not included.",
    }
