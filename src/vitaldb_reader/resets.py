"""Put the records written around a SNUADC clock reset back in the order they were sampled.

When the SNUADC's sample clock has drifted about half a second ahead of the recording PC, the recorder
pulls the track back (a *reset*). At every reset the five records after it are written to the file out
of order. With ``p`` the last record before the reset and ``D`` one record's duration at the nominal
rate (0.2 s), two of the five records continue the old grid (stamped ``p + D`` and ``p + 2D``) and three
start a new grid (``c0``, ``c0 + D`` and ``c0 + 2D``, with ``p < c0 < p + D``). The usual file order is::

    c0 = p + a (new)   c1 = p + D (old)   c2 = c0 + D (new)   c3 = p + 2D (old)   c4 = c0 + 2D (new)

Read in file order, or sorted by stamp, every junction inside the cluster breaks the waveform. Put the
old-grid records first and the new-grid records after them, each in file order (here ``c1, c3, c0, c2,
c4``), and every junction is as smooth as an ordinary record boundary, no samples are missing at any of
them, and the stamps make one clean step back: the old grid runs on for two more records, then the new
grid starts. Found on 315 resets in step-mode SNUADC files; checked on 392 resets in 174 step-mode
cases, in every SNUADC channel (the same clusters in each), with 9 more resets already written in order
(VitalDB timing graph document, 17 October).

About one reset in ten interleaves differently (old-grid records at file positions 2 and 3, or 1 and 2)
and stamps one record with its neighbour's time, so two records share a stamp. The same rule orders
those, and the record stamped twice gets the time its grid gives it; :attr:`ResetOrder.restamped`
counts them. A record whose stamp sits on neither grid, or a cluster with other than two old-grid and
three new-grid records, is left in file order: matched on the stamps alone, the rule never touches a
stretch that merely resembles a reset in the signal.

Two things this module does not change. Most resets also hold one extra sample that reads 0 in every
channel, a filler with no time of its own, a little after the reset; and the about 20 ms of signal the
reset loses goes missing inside the last old-grid record, at no fixed sample. Both are left as stored.

:func:`read_vital <vitaldb_reader.vital_file.read_vital>` keeps file
order, as ground truth; this module is the importer's decision on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from vitaldb_reader.vital_file import TRACK_WAVE, VitalCase, VitalTrack


__all__ = ["RULE_VERSION", "ResetOrder", "find_reset_clusters", "reorder_case", "reorder_track"]

RULE_VERSION = "snuadc-reset-order-2026.10.17"
"""Identifies this rule in the metadata of everything read with it."""

_DEVICE = "SNUADC"
_CLUSTER = 5
_OLD = 2  # records of a cluster that continue the old grid
_DUPLICATE = 2  # a repeated stamp is shared by two records


@dataclass(frozen=True)
class ResetOrder:
    """What :func:`reorder_track` did to one track."""

    clusters: int
    """Reset clusters put back in sampling order."""
    first_records: tuple[int, ...]
    """File index of ``c0`` of each reordered cluster."""
    restamped: int = 0
    """Records whose duplicated stamp was replaced by the time their grid gives them."""


@dataclass(frozen=True)
class _Cluster:
    first: int
    """File index of ``c0``."""
    order: tuple[int, ...]
    """File indices in sampling order."""
    times: tuple[float, ...]
    """The stamp of each record in :attr:`order`, on its grid."""


def _match(times: np.ndarray, r: int, d: float, half: float) -> _Cluster | None:
    p, c0 = float(times[r - 1]), float(times[r])
    if not half < c0 - p < d - half:
        return None
    old: list[int] = []
    new: list[int] = []
    for k in range(r, r + _CLUSTER):
        on_old = abs((float(times[k]) - p + d / 2) % d - d / 2) < half
        on_new = abs((float(times[k]) - c0 + d / 2) % d - d / 2) < half
        if on_old == on_new:
            return None
        (old if on_old else new).append(k)
    if len(old) != _OLD or new[0] != r:
        return None
    grid = [p + (j + 1) * d for j in range(len(old))] + [c0 + j * d for j in range(len(new))]
    order = old + new
    stamps = times[r - 1 : r + _CLUSTER]
    for k, g in zip(order, grid, strict=True):
        # every stamp sits on its grid, or repeats the stamp of another record of the cluster
        if abs(float(times[k]) - g) >= half and np.sum(np.abs(stamps - times[k]) < half) < _DUPLICATE:
            return None
    return _Cluster(first=r, order=tuple(order), times=tuple(grid))


def find_reset_clusters(times: np.ndarray, sizes: np.ndarray, srate: float) -> list[int]:
    """Find the reset clusters in one waveform's records.

    Args:
        times: Record stamps, in file order.
        sizes: Samples per record.
        srate: The nominal sample rate.

    Returns:
        The file index of ``c0`` of every cluster that matches the rule, in file order.
    """
    return [c.first for c in _clusters(times, sizes, srate)]


def _clusters(times: np.ndarray, sizes: np.ndarray, srate: float) -> list[_Cluster]:
    n = times.size
    if n < _CLUSTER + 1 or srate <= 0:
        return []
    half = 0.5 / srate
    found: list[_Cluster] = []
    back = np.flatnonzero(np.diff(times) - sizes[:-1] / srate < -half)  # stamped before the previous record ends
    for i in back:
        r = int(i) + 1  # candidate c0
        if r + _CLUSTER > n or (found and r < found[-1].first + _CLUSTER):
            continue
        # the grids are set by the records before c4, so c4 itself may hold a sample more or less
        block = sizes[r - 1 : r + _CLUSTER - 1]
        if np.any(block != block[0]) or block[0] <= 0:
            continue
        cluster = _match(times, r, float(block[0]) / srate, half)
        if cluster is not None:
            found.append(cluster)
    return found


def reorder_track(track: VitalTrack) -> tuple[VitalTrack, ResetOrder]:
    """Return a SNUADC waveform with its reset clusters in sampling order.

    Args:
        track: Any track. Only SNUADC waveforms are changed.

    Returns:
        The track (a new object if anything moved, else the same one) and what was done.
    """
    if track.device != _DEVICE or track.track_type != TRACK_WAVE or not track.record_times.size:
        return track, ResetOrder(0, ())
    sizes = track.record_sizes[: track.record_times.size]
    clusters = _clusters(track.record_times, sizes, float(track.srate))
    if not clusters:
        return track, ResetOrder(0, ())
    order = np.arange(track.record_times.size)
    times = track.record_times.copy()
    for c in clusters:
        r = c.first
        order[r : r + _CLUSTER] = c.order
        grid = np.asarray(c.times)
        times[r : r + _CLUSTER] = np.where(
            np.abs(track.record_times[list(c.order)] - grid) < 0.5 / float(track.srate),
            track.record_times[list(c.order)],
            grid,
        )
    offsets = track.record_offsets
    sample_index = np.arange(offsets[-1])
    for c in clusters:
        r = c.first
        sample_index[offsets[r] : offsets[r + _CLUSTER]] = np.concatenate(
            [np.arange(offsets[k], offsets[k + 1]) for k in c.order]
        )
    values = np.asarray(track.values)[sample_index]
    moved = replace(track, record_times=times, record_sizes=track.record_sizes[order], values=values)
    restamped = int(np.sum(times != track.record_times[order]))
    return moved, ResetOrder(len(clusters), tuple(c.first for c in clusters), restamped)


def reorder_case(case: VitalCase) -> tuple[VitalCase, dict[str, ResetOrder]]:
    """Apply :func:`reorder_track` to every track of a case.

    Args:
        case: A case as :func:`read_vital` returns it.

    Returns:
        The case with its SNUADC waveforms reordered, and per changed track what was done.
    """
    tracks: dict[str, VitalTrack] = {}
    done: dict[str, ResetOrder] = {}
    for name, track in case.tracks.items():
        tracks[name], result = reorder_track(track)
        if result.clusters:
            done[name] = result
    if not done:
        return case, done
    return replace(case, tracks=tracks), done
