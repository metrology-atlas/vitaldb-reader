"""Tests of the SNUADC reset-cluster reordering, on synthetic records shaped like the release's."""

import numpy as np
import pytest

from vitaldb_reader.resets import find_reset_clusters, reorder_track
from vitaldb_reader.vital_file import TRACK_WAVE, VitalTrack


FS = 500.0
N = 100  # samples per record
D = N / FS


def _track(file_order, stamps, device="SNUADC", extra_last=0):
    """A track whose sample values count up in sampling order, written in ``file_order``."""
    sizes = np.full(len(file_order), N, dtype=np.int64)
    if extra_last:
        sizes[file_order.index(max(file_order))] += extra_last
    starts = np.r_[0, np.cumsum([sizes[file_order.index(k)] for k in range(len(file_order))])]
    values = np.concatenate([np.arange(starts[k], starts[k] + sizes[i]) for i, k in enumerate(file_order)])
    return VitalTrack(
        name="ECG_II",
        device=device,
        track_type=TRACK_WAVE,
        fmt=6,
        unit="mV",
        srate=FS,
        gain=1.0,
        offset=0.0,
        montype=0,
        mindisp=0.0,
        maxdisp=1.0,
        record_times=np.asarray(stamps, dtype=float),
        record_sizes=sizes,
        values=values.astype(np.uint16),
    )


def _reset(order_in_file, cluster_stamps, before=10, after=10, a=0.1193):
    """Records 0..before-1 on the old grid, a 5-record cluster, then the new grid.

    ``order_in_file`` lists the sampling positions (0-4) of the cluster's records in file order;
    ``cluster_stamps`` their stamps relative to the last record before the reset.
    """
    t0 = 1000.0
    file_order = list(range(before)) + [before + k for k in order_in_file]
    stamps = [t0 + i * D for i in range(before)]
    p = stamps[-1]
    stamps += [p + s for s in cluster_stamps]
    new0 = p + a
    for j in range(after):
        file_order.append(before + 5 + j)
        stamps.append(new0 + (3 + j) * D)
    return file_order, stamps


STANDARD = ([2, 0, 3, 1, 4], [0.1193, D, 0.1193 + D, 2 * D, 0.1193 + 2 * D])


def test_standard_cluster_is_put_in_sampling_order():
    track = _track(*_reset(*STANDARD))
    assert find_reset_clusters(track.record_times, track.record_sizes, FS) == [10]
    moved, done = reorder_track(track)
    assert done.clusters == 1 and done.restamped == 0
    assert np.all(np.diff(moved.values.astype(int)) == 1), "samples must run in sampling order"
    steps = np.diff(moved.record_times) - D
    assert np.sum(steps < -1e-3) == 1 and steps.min() == pytest.approx(0.1193 - 3 * D)
    assert np.flatnonzero(steps < -1e-3)[0] == 11, "the step back comes after the two old-grid records"


@pytest.mark.parametrize(
    ("order", "stamps"),
    [
        # old-grid records at file positions 2 and 3; the second new-grid record repeats c0's stamp
        ([2, 3, 0, 1, 4], [0.12, 0.12, D, 2 * D, 0.12 + 2 * D]),
        # old-grid records at file positions 1 and 2; the second repeats the first's stamp
        ([2, 0, 1, 3, 4], [0.119, D, D, 0.119 + D, 0.119 + 2 * D]),
    ],
)
def test_variant_clusters_are_ordered_and_restamped(order, stamps):
    track = _track(*_reset(order, stamps, a=stamps[0]))
    moved, done = reorder_track(track)
    assert done.clusters == 1 and done.restamped == 1
    assert np.all(np.diff(moved.values.astype(int)) == 1)
    steps = np.diff(moved.record_times) - D
    assert np.sum(np.abs(steps) > 1e-3) == 1, "after restamping only the reset step remains"


def test_last_record_may_hold_an_extra_sample():
    track = _track(*_reset(*STANDARD), extra_last=0)
    sizes = track.record_sizes.copy()
    sizes[14] += 1  # c4 in file order holds 101 samples
    assert find_reset_clusters(track.record_times, sizes, FS) == [10]


def test_other_devices_and_lookalikes_are_left_alone():
    track = _track(*_reset(*STANDARD), device="Primus")
    same, done = reorder_track(track)
    assert same is track and done.clusters == 0
    order, stamps = STANDARD
    broken = list(stamps)
    broken[3] += 0.05  # an old-grid record off both grids
    track = _track(*_reset(order, broken))
    same, done = reorder_track(track)
    assert same is track and done.clusters == 0


def test_a_clean_step_needs_no_reordering():
    file_order = list(range(20))
    stamps = [1000.0 + i * D for i in range(10)] + [1000.0 + 9 * D + 0.1193 + i * D for i in range(1, 11)]
    stamps = [s if i < 10 else s - 0.48 for i, s in enumerate(stamps)]
    track = _track(file_order, stamps)
    same, done = reorder_track(track)
    assert same is track and done.clusters == 0


