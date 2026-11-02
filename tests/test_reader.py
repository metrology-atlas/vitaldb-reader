"""Tests of the reader, timing, lattice, resolution and timing model, on a synthetic fixture (no real data)."""

from fractions import Fraction
import itertools
from pathlib import Path

import numpy as np
import pytest

from vitaldb_reader.errors import VitalFormatError
from vitaldb_reader.lattice import decode, find_lattice
from vitaldb_reader.resolution import _runs, near_duplicates, track_resolution
from vitaldb_reader.timing import TimingMode, nominal_rate, runs, segments
from vitaldb_reader.vital_file import TRACK_STRING, TRACK_WAVE, read_vital


FIXTURES = Path(__file__).parent / "fixtures"
VITAL = FIXTURES / "vital_files" / "0001.vital"


@pytest.fixture(scope="module")
def case():
    return read_vital(VITAL)


# --- reader -------------------------------------------------------------------------------------------


def test_reader_keeps_native_dtypes_and_calibration(case):
    ecg = case.tracks["SNUADC/ECG_II"]
    assert ecg.track_type == TRACK_WAVE
    assert ecg.values.dtype == np.dtype("<u2")
    assert ecg.gain == pytest.approx(0.0098745703125) and ecg.offset == pytest.approx(-4.95626)
    assert case.tracks["BIS/EEG1_WAV"].values.dtype == np.dtype("<f4")
    assert case.tracks["EVENT"].device == "" and case.tracks["EVENT"].track_type == TRACK_STRING
    assert case.devices["Solar8000"].type_name == "Solar 8000M"


def test_reader_keeps_file_order_and_repeated_stamps(case):
    art = case.tracks["SNUADC/ART"]
    assert np.any(np.diff(art.record_times) < 0), "the fixture's step back in time must survive"
    hr = case.tracks["Solar8000/HR"]
    assert np.any(np.diff(hr.record_times) == 0)


def test_reader_keeps_a_name_declared_twice_apart(case):
    first, second = case.tracks["Solar8000/BT"], case.tracks["Solar8000/BT#2"]
    assert first.record_times.size == 30
    assert second.record_times.size == 5
    assert second.values[0] == np.float32(35.0)


def test_reader_rejects_non_vital(tmp_path):
    bad = tmp_path / "x.vital"
    bad.write_bytes(b"not gzip")
    with pytest.raises(VitalFormatError):
        read_vital(bad)


def test_packed_and_unpacked_read_the_same(tmp_path, case):
    vitaldb = pytest.importorskip("vitaldb")
    vf = vitaldb.VitalFile(str(VITAL))
    vf.to_vital(str(tmp_path / "packed.vital"), packed=True)
    packed = read_vital(tmp_path / "packed.vital")
    # tracks without overlaps or steps back: the packed copy's gap fill is cut out again
    for name in ("BIS/EEG1_WAV", "Primus/AWP"):
        assert np.asarray(packed.tracks[name].values).tobytes() == case.tracks[name].values.tobytes()


# --- timing -------------------------------------------------------------------------------------------


def test_nominal_rate_reads_the_decimal_spelling(case):
    assert nominal_rate(case.tracks["Primus/AWP"]) == Fraction(125, 2)
    assert nominal_rate(case.tracks["SNUADC/ART"]) == 500


def test_runs_split_at_gap_overlap_and_step_back(case):
    assert len(runs(case.tracks["SNUADC/ECG_II"])) == 3  # gap, then overlap
    assert len(runs(case.tracks["SNUADC/ART"])) >= 3  # the step back opens and closes a run
    assert len(runs(case.tracks["BIS/EEG1_WAV"])) == 1


@pytest.mark.parametrize("mode", list(TimingMode))
def test_segments_cover_every_sample_once(case, mode):
    track = case.tracks["SNUADC/ECG_II"]
    segs = segments(track, case.origin, mode)
    assert segs[0].sample_start == 0 and segs[-1].sample_stop == track.values.size
    assert all(a.sample_stop == b.sample_start for a, b in itertools.pairwise(segs))


def test_file_timing_reproduces_drifting_stamps_within_tolerance(case):
    track = case.tracks["SNUADC/ECG_II"]
    for seg in segments(track, case.origin, TimingMode.FILE, tolerance_s=1e-6):
        assert seg.max_error_s <= 1.1e-6
        # the release's stamps run 2**-20 fast relative to 500 Hz; the fitted period says so exactly
        if seg.record_stop - seg.record_start > 10:
            assert abs(float(seg.rate_hz / 500 - 1) - 2**-20) < 1e-9


def test_nominal_timing_reports_its_drift(case):
    segs = segments(case.tracks["SNUADC/ECG_II"], case.origin, TimingMode.NOMINAL)
    assert all(s.period_s == Fraction(1, 500) for s in segs)
    assert max(s.max_error_s for s in segs) > 5e-6


# --- lattice ------------------------------------------------------------------------------------------


def test_lattice_is_bit_exact_or_absent():
    rng = np.random.default_rng(0)
    eeg = (rng.integers(-20000, 20000, 5000) * 0.01).astype(np.float32)
    found = find_lattice(eeg)
    assert found is not None and found.step == 0.01  # noqa: RUF069 - the step must be exactly the decimal 0.01
    assert decode(found.k, found.step, found.offset, eeg.dtype).tobytes() == eeg.tobytes()
    assert find_lattice(rng.normal(size=100).astype(np.float32)) is None
    assert find_lattice(np.array([1.0, np.nan], dtype=np.float32)) is None


# --- timing, resolution and timing model ----------------------------------------------------------------------------------------


def test_resync_spreads_the_jump_over_the_run(case):
    from vitaldb_reader.timing import ResyncLimits  # noqa: PLC0415

    track = case.tracks["SNUADC/ECG_II"]
    # the fixture's second run (20 s) ends in a 40-sample overlap: 4000 ppm, so widen the limits
    limits = ResyncLimits(min_run_s=10, max_ppm=5000)
    segs = segments(track, case.origin, TimingMode.RESYNC, limits=limits)
    run2 = segs[1]
    assert run2.rate_source == "resync"
    # the corrected run ends exactly where the next run starts
    end = run2.start_s + run2.n_samples * run2.period_s
    assert end == segs[2].start_s
    assert run2.ppm > 0  # an overlap means the device ran fast
    # a run that ends in a real gap (2.5 s) borrows the track's corrected rate
    assert segs[0].rate_source == "track" and segs[0].period_s == run2.period_s
    # with default limits nothing qualifies, and every run is nominal
    assert {s.rate_source for s in segments(track, case.origin, TimingMode.RESYNC)} == {"nominal"}


# --- value resolution and duplicates -----------------------------------------------------------------


def test_resolution_of_integer_waveform_and_numeric(case):
    wave = next(t for t in case.tracks.values() if t.track_type == TRACK_WAVE and t.is_integer_format)
    res = track_resolution(wave, case.origin)
    assert res is not None and res.step == wave.gain and res.step_source == "gain"
    assert res.converter_bits is not None and res.converter_range is not None
    hi = int(np.asarray(wave.values).max())
    assert 2**res.converter_bits - 1 >= hi > 2 ** (res.converter_bits - 1) - 1
    assert res.converter_range[0] == wave.offset
    numeric = next(t for t in case.tracks.values() if t.track_type not in {TRACK_WAVE, TRACK_STRING})
    rn = track_resolution(numeric, case.origin)
    assert rn is not None and rn.step is not None and rn.step > 0


def test_limit_runs_found_at_converter_limits():
    x = np.array([5] * 10 + [1023] * 30 + [7] * 5 + [0] * 3)
    assert _runs(x, 1023, 25) == [(10, 40)]
    assert _runs(x, 0, 25) == []


def test_repeats_dropped_only_when_close_and_same_value():
    t = np.array([0, 1000, 2_000_000, 2_001_000, 4_000_000])
    v = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    assert near_duplicates(t, v, 50_000).tolist() == [True, False, True, True, True]


# --- timing model -------------------------------------------------------------------------------------


def _row(**kw):
    # adm on a minute: the case starts at 08:00 (adm = -8 h); EMR times share the seconds phase 0
    base = {"adm": str(-8 * 3600), "anestart": "300", "aneend": "3600", "opstart": "900", "opend": "3000"}
    return {**base, **kw}


def test_clock_minute_needs_adm_and_a_shared_phase():
    from vitaldb_reader.timing_model import case_clock_minute  # noqa: PLC0415

    assert case_clock_minute(_row()) == 8 * 3600
    assert case_clock_minute(_row(adm="-28830")) is None
    assert case_clock_minute(_row(anestart="301", aneend="3602", opstart="903")) is None


def test_five_minute_entries_are_rounded_up_and_minute_entries_are_not():
    from vitaldb_reader.timing_model import clinical_time_resolution  # noqa: PLC0415

    five = clinical_time_resolution(_row(), "anestart")
    assert five is not None
    assert five["grid"] == "5 min" and five["rounding"] == "up" and five["basis"] == "measured"
    assert five["interval_s"] == [0.0, 300.0] and five["event_estimate_s"] == pytest.approx(150.0)
    # rounded up, so the event could be known at the recorded time itself
    assert five["available_s"] == pytest.approx(300.0)
    op = clinical_time_resolution(_row(), "opstart")
    assert op is not None and op["basis"].startswith("assumed")
    minute = clinical_time_resolution(_row(anestart="360"), "anestart")
    assert minute is not None and minute["grid"] == "minute" and minute["interval_s"] == [300.0, 420.0]
    # rounding direction unknown: the event may lie up to a minute after the recorded time
    assert minute["available_s"] == pytest.approx(420.0)
    unknown = clinical_time_resolution(_row(adm="-28830"), "aneend")
    assert unknown is not None and unknown["grid"] == "unknown" and unknown["event_estimate_s"] is None
    assert clinical_time_resolution(_row(), "casestart") is None
    start = clinical_time_resolution({"casestart": "0"}, "casestart")
    assert start is not None and start["sd_s"] == pytest.approx(0.0) and start["available_s"] == pytest.approx(0.0)
    end = clinical_time_resolution({"caseend": "100"}, "caseend")
    assert end is not None and end["available_s"] == pytest.approx(100.5)
    assert clinical_time_resolution(_row(anestart=""), "anestart") is None


def test_numeric_lag_table_is_consistent():
    from vitaldb_reader.timing_model import (  # noqa: PLC0415
        MODEL_VERSION,
        NUMERIC_LAG,
        numeric_lag,
    )

    hr = numeric_lag("Solar8000/HR")
    assert hr is not None and len(hr.modes) == 2
    assert abs(sum(share for _, share in hr.modes) - 1.0) < 1e-9
    assert numeric_lag("Solar8000/BT") is None
    bis = numeric_lag("BIS/BIS")
    assert bis is not None and bis.basis == "measured" and bis.window_s >= 15
    for lag in NUMERIC_LAG.values():
        lo, mid, hi = lag.delay_from_end_s
        assert (lo is None or lo <= mid) and (hi is None or mid <= hi)
        assert lag.window_s >= 0
        meta = lag.metadata()
        assert meta["model_version"] == MODEL_VERSION
        # every timing quantity is anchored at the window's end; the centre is only a derived description
        assert [k for k in meta if "centre" in k or "center" in k] == ["support_centre_s"]
        assert meta["support_centre_s"] == pytest.approx(mid + lag.window_s / 2, abs=0.01)
        # a delay below zero breaks the availability rule, so the entry must say it is provisional
        if (lo is not None and lo < 0) or mid < 0:
            assert lag.provisional


