"""Read a VitalDB ``.vital`` file into a neutral, native-resolution form.

This module is the ground truth that every VitalDB importer is checked against. It reads each case once,
keeps every track at the resolution and in the dtype the file stores it, and resamples nothing. It does
not use ``vitaldb.VitalFile.to_numpy`` or ``vital_recs(..., interval=...)``: both put every track on one
shared grid, which erases the native timing, the gaps and the irregular times of the numeric tracks.

The ``.vital`` container is a gzip stream. After a ``VITA`` signature, a format version and a short
header (time-zone offset, and in later versions the file's start and end times and a "packed" flag), it
is a sequence of packets: one byte of type, four bytes of length, then the body. The packets this
reader understands are:

- ``9`` device information: an id, a type name, a display name and optionally a port;
- ``0`` track information: an id, the track type (1 waveform, 2 numeric, 5 string), the storage format
  of its values, its name and unit, display limits, colour, sample rate, gain, offset, monitor type and
  the id of its device;
- ``1`` a record: a time stamp (float64 seconds), a track id, and the values. A waveform record holds a
  block of samples; sample ``j`` of it sits at ``t + j / srate``. A numeric record holds one value and a
  string record holds one string;
- ``6`` a command: ``5`` sets the display order of the tracks, ``6`` clears the event track.

Physical values of an integer-format waveform are ``value * gain + offset``. Float formats are stored as
physical values already. Records are kept in file order, never sorted, so an importer can see and decide
what to do with a record that goes back in time.

A "packed" file (written by newer tools) stores each waveform track as one record and fills gaps with a
sentinel value. :func:`read_vital` cuts those sentinel runs back out, so a packed and an unpacked copy of
the same recording read the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
from pathlib import Path
import struct

import numpy as np

from vitaldb_reader.errors import VitalFormatError


__all__ = [
    "FORMAT_DTYPES",
    "TRACK_NUMERIC",
    "TRACK_STRING",
    "TRACK_WAVE",
    "VitalCase",
    "VitalDevice",
    "VitalTrack",
    "read_vital",
]

# Track types, as the file numbers them.
TRACK_WAVE = 1
TRACK_NUMERIC = 2
TRACK_STRING = 5

# Storage format code -> little-endian NumPy dtype.
FORMAT_DTYPES: dict[int, np.dtype] = {
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
    3: np.dtype("<i1"),
    4: np.dtype("<u1"),
    5: np.dtype("<i2"),
    6: np.dtype("<u2"),
    7: np.dtype("<i4"),
    8: np.dtype("<u4"),
}

# Formats 1 (float32) and 2 (float64) store physical values; every higher code is an integer format.
_FLOAT_FORMATS = frozenset({1, 2})
_MAX_FLOAT_FORMAT = 2

# The value a packed file writes into a gap, per integer format. Float formats use NaN.
_PACKED_GAP: dict[int, int] = {3: -128, 4: 255, 5: -32768, 6: 65535, 7: -2147483648, 8: 4294967295}

# Header layout: bytes 0-1 hold the time-zone offset, 10-17 and 18-25 the start and end times, and
# byte 26 the "packed" flag. Each field is present only when the header is long enough to hold it.
_HEADER_DGMT_END = 2
_HEADER_TIMES_END = 26
_HEADER_PACKED_END = 27
# A record packet's fixed part: info length (u16), time (f64) and track id (u16).
_RECORD_FIXED_LEN = 12

_PACKET_TRACK_INFO = 0
_PACKET_RECORD = 1
_PACKET_COMMAND = 6
_PACKET_DEVICE_INFO = 9
_COMMAND_TRACK_ORDER = 5
_COMMAND_RESET_EVENTS = 6

_U16 = struct.Struct("<H").unpack_from
_I16 = struct.Struct("<h").unpack_from
_U32 = struct.Struct("<L").unpack_from
_F32 = struct.Struct("<f").unpack_from
_F64 = struct.Struct("<d").unpack_from


@dataclass(frozen=True)
class VitalDevice:
    """One device as the file declares it."""

    name: str
    """The name tracks are qualified with, as in ``SNUADC`` of ``SNUADC/ECG_II``."""
    type_name: str
    """The device model the recorder reported, such as ``Solar 8000M`` for the device named ``Solar8000``."""
    port: str
    """The port the device was attached to, or an empty string."""


@dataclass
class VitalTrack:
    """One track at native resolution: its declaration and every record, in file order."""

    name: str
    """The track name within its device, such as ``ECG_II``."""
    device: str
    """The device name, or an empty string for a track with no device (the ``EVENT`` track)."""
    track_type: int
    """:data:`TRACK_WAVE`, :data:`TRACK_NUMERIC` or :data:`TRACK_STRING`."""
    fmt: int
    """The storage format code, a key of :data:`FORMAT_DTYPES` for waveform and numeric tracks."""
    unit: str
    srate: float
    """The nominal sample rate the file states, in Hz, as stored (a float32). Zero for non-waveforms."""
    gain: float
    offset: float
    montype: int
    mindisp: float
    maxdisp: float
    record_times: np.ndarray
    """One float64 time stamp per record, in seconds, exactly as stored."""
    record_sizes: np.ndarray
    """Samples per record (int64). All ones for numeric and string tracks."""
    values: np.ndarray | list[str]
    """Every value in record order, in the stored dtype. A list of strings for a string track."""
    record_offsets: np.ndarray = field(init=False)
    """Index of each record's first value in :attr:`values` (int64, one longer than the records)."""

    def __post_init__(self) -> None:
        """Derive the record offsets from the record sizes."""
        self.record_offsets = np.concatenate(([0], np.cumsum(self.record_sizes, dtype=np.int64)))

    @property
    def dtname(self) -> str:
        """The qualified name, ``device/track``, or the bare name for a track with no device."""
        return f"{self.device}/{self.name}" if self.device else self.name

    @property
    def dtype(self) -> np.dtype | None:
        """The stored value dtype, or ``None`` for a string track."""
        return None if self.track_type == TRACK_STRING else FORMAT_DTYPES[self.fmt]

    @property
    def is_integer_format(self) -> bool:
        """Whether values are stored as integer counts that ``gain`` and ``offset`` scale."""
        return self.track_type != TRACK_STRING and self.fmt > _MAX_FLOAT_FORMAT

    def physical(self, values: np.ndarray | None = None) -> np.ndarray:
        """Return physical values as float64: ``value * gain + offset`` for integer formats.

        Args:
            values: A slice of :attr:`values`, or ``None`` for all of them.

        Returns:
            Float64 physical values. Float formats come back unchanged apart from the widening.
        """
        raw = np.asarray(self.values if values is None else values)
        if self.is_integer_format:
            return raw.astype(np.float64) * self.gain + self.offset
        return raw.astype(np.float64)


@dataclass
class VitalCase:
    """Everything one ``.vital`` file holds, at native resolution."""

    path: Path
    version: int
    dgmt: int | None
    """Minutes the recorder's clock was from UTC (``UTC - local``), if the header states it."""
    header_start: float
    header_end: float
    packed: bool
    devices: dict[str, VitalDevice]
    tracks: dict[str, VitalTrack]
    """Tracks keyed by :attr:`VitalTrack.dtname`, in the order the file declares them."""
    display_order: list[str]
    unknown_packets: dict[int, int]
    """Counts of packet types this reader skipped."""

    @property
    def origin(self) -> float:
        """The earliest record time stamp of any track: the case's relative zero.

        Raises:
            VitalFormatError: If no track holds a record.
        """
        firsts = [float(t.record_times.min()) for t in self.tracks.values() if t.record_times.size]
        if not firsts:
            raise VitalFormatError(f"{self.path}: holds no records")
        return min(firsts)


def _read_str(buf: bytes | memoryview, pos: int, end: int) -> tuple[str, int]:
    if pos + 4 > end:
        raise VitalFormatError("string length runs past the end of its packet")
    length = _U32(buf, pos)[0]
    pos += 4
    if pos + length > end:
        raise VitalFormatError("string runs past the end of its packet")
    return bytes(buf[pos : pos + length]).decode("utf-8", "replace"), pos + length


@dataclass
class _TrackBuilder:
    """Accumulates one track's records while the file streams past."""

    name: str
    device: str
    track_type: int
    fmt: int
    unit: str
    srate: float
    gain: float
    offset: float
    montype: int
    mindisp: float
    maxdisp: float
    skip: bool = False
    """Whether the caller asked to leave this track out: its records are not kept."""
    times: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    chunks: list[bytes] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)

    def build(self, packed: bool) -> VitalTrack:
        times = np.asarray(self.times, dtype=np.float64)
        sizes = np.asarray(self.sizes, dtype=np.int64)
        values: np.ndarray | list[str]
        if self.track_type == TRACK_STRING:
            values = self.strings
        else:
            values = np.frombuffer(b"".join(self.chunks), dtype=FORMAT_DTYPES[self.fmt]).copy()
            if packed and self.track_type == TRACK_WAVE:
                times, sizes, values = _unpack_gaps(times, sizes, values, self.fmt, float(self.srate))
        return VitalTrack(
            name=self.name,
            device=self.device,
            track_type=self.track_type,
            fmt=self.fmt,
            unit=self.unit,
            srate=self.srate,
            gain=self.gain,
            offset=self.offset,
            montype=self.montype,
            mindisp=self.mindisp,
            maxdisp=self.maxdisp,
            record_times=times,
            record_sizes=sizes,
            values=values,
        )


def _unpack_gaps(
    times: np.ndarray, sizes: np.ndarray, values: np.ndarray, fmt: int, srate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the gap fill out of a packed waveform track.

    Returns:
        The record times, record sizes and values, with one record per gap-free run.
    """
    if srate <= 0 or values.size == 0:
        return times, sizes, values
    gap = np.isnan(values) if fmt in _FLOAT_FORMATS else values == _PACKED_GAP[fmt]
    if not gap.any():
        return times, sizes, values
    new_times, new_sizes, keep = [], [], []
    for t0, start, stop in zip(times, np.r_[0, np.cumsum(sizes)[:-1]], np.cumsum(sizes), strict=True):
        g = np.r_[True, gap[start:stop], True]
        edges = np.flatnonzero(np.diff(g.astype(np.int8)))
        for a, b in zip(edges[0::2], edges[1::2], strict=True):
            new_times.append(t0 + a / srate)
            new_sizes.append(b - a)
            keep.append(np.arange(start + a, start + b))
    idx = np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)
    return np.asarray(new_times), np.asarray(new_sizes, dtype=np.int64), values[idx]


@dataclass(frozen=True)
class _Header:
    """The file's signature-checked header fields."""

    version: int
    length: int
    dgmt: int | None
    start: float
    end: float
    packed: bool


def _read_header(path: Path, data: bytes) -> _Header:
    """Check the signature and read the header.

    Returns:
        The header fields.

    Raises:
        VitalFormatError: If the ``VITA`` signature is missing.
    """
    if data[:4] != b"VITA":
        raise VitalFormatError(f"{path}: missing the VITA signature")
    version = _U32(data, 4)[0]
    header_len = _U16(data, 8)[0]
    header = data[10 : 10 + header_len]
    return _Header(
        version=version,
        length=header_len,
        dgmt=_I16(header, 0)[0] if header_len >= _HEADER_DGMT_END else None,
        start=_F64(header, 10)[0] if header_len >= _HEADER_TIMES_END else 0.0,
        end=_F64(header, 18)[0] if header_len >= _HEADER_TIMES_END else 0.0,
        packed=header_len >= _HEADER_PACKED_END and header[26] == 1,
    )


class _PacketReader:
    """Walks the packets of one file, collecting devices, track builders and the display order."""

    def __init__(self, path: Path, data: bytes, track_names: set[str] | None) -> None:
        self.path = path
        self.data = data
        self.view = memoryview(data)
        self.track_names = track_names
        self.devices_by_id: dict[int, VitalDevice] = {}
        self.builders: dict[int, _TrackBuilder] = {}
        self.order_ids: list[int] = []
        self.unknown: dict[int, int] = {}

    def read(self, pos: int) -> None:
        """Read every packet from byte ``pos`` to the end of the file.

        Raises:
            VitalFormatError: If a packet header or body runs past the end of the file.
        """
        data, n = self.data, len(self.data)
        while pos < n:
            if pos + 5 > n:
                raise VitalFormatError(f"{self.path}: truncated packet header at byte {pos}")
            ptype, plen = data[pos], _U32(data, pos + 1)[0]
            body, end = pos + 5, pos + 5 + plen
            if end > n:
                raise VitalFormatError(f"{self.path}: packet at byte {pos} runs past the end of the file")
            pos = end
            if ptype == _PACKET_RECORD:
                self._record(body, end, plen)
            elif ptype == _PACKET_TRACK_INFO:
                self._track_info(body, end)
            elif ptype == _PACKET_DEVICE_INFO:
                self._device_info(body, end)
            elif ptype == _PACKET_COMMAND:
                self._command(body)
            else:
                self.unknown[ptype] = self.unknown.get(ptype, 0) + 1

    def _record(self, body: int, end: int, plen: int) -> None:
        """Append one record to its track's builder.

        Raises:
            VitalFormatError: If the record names an undeclared track or overruns its packet.
        """
        if plen < _RECORD_FIXED_LEN:
            return
        data, view, path = self.data, self.view, self.path
        info_len = _U16(data, body)[0]
        time = _F64(data, body + 2)[0]
        tid = _U16(data, body + 10)[0]
        builder = self.builders.get(tid)
        if builder is None:
            raise VitalFormatError(f"{path}: record for undeclared track id {tid}")
        if builder.skip:
            return
        p = body + 2 + info_len
        track_type = builder.track_type
        if track_type == TRACK_WAVE:
            count = _U32(data, p)[0]
            width = FORMAT_DTYPES[builder.fmt].itemsize
            if p + 4 + count * width > end:
                raise VitalFormatError(f"{path}: waveform record overruns its packet")
            builder.chunks.append(bytes(view[p + 4 : p + 4 + count * width]))
            builder.sizes.append(count)
        elif track_type == TRACK_NUMERIC:
            width = FORMAT_DTYPES[builder.fmt].itemsize
            if p + width > end:
                raise VitalFormatError(f"{path}: numeric record overruns its packet")
            builder.chunks.append(bytes(view[p : p + width]))
            builder.sizes.append(1)
        else:
            text, _ = _read_str(view, p + 4, end)
            builder.strings.append(text)
            builder.sizes.append(1)
        builder.times.append(time)

    def _track_info(self, body: int, end: int) -> None:
        """Declare one track.

        Raises:
            VitalFormatError: If the track's type or storage format is unknown.
        """
        data, view = self.data, self.view
        tid, track_type, fmt = _U16(data, body)[0], data[body + 2], data[body + 3]
        name, p = _read_str(view, body + 4, end)
        unit = ""
        numbers: dict[str, float] = {"mindisp": 0.0, "maxdisp": 0.0, "srate": 0.0, "gain": 1.0, "offset": 0.0}
        montype = 0
        did = 0
        # Later fields are optional: older files stop early. Read each one only if it fits.
        if p < end:
            unit, p = _read_str(view, p, end)
        for key, unpack, width in (
            ("mindisp", _F32, 4),
            ("maxdisp", _F32, 4),
            ("_col", _U32, 4),
            ("srate", _F32, 4),
            ("gain", _F64, 8),
            ("offset", _F64, 8),
        ):
            if p + width <= end:
                numbers[key] = unpack(data, p)[0]
                p += width
        if p + 1 <= end:
            montype = data[p]
            p += 1
        if p + 4 <= end:
            did = _U32(data, p)[0]
        if track_type in {TRACK_WAVE, TRACK_NUMERIC} and fmt not in FORMAT_DTYPES:
            raise VitalFormatError(f"{self.path}: track {name!r} has unknown storage format {fmt}")
        if track_type not in {TRACK_WAVE, TRACK_NUMERIC, TRACK_STRING}:
            raise VitalFormatError(f"{self.path}: track {name!r} has unknown track type {track_type}")
        device = self.devices_by_id[did].name if did in self.devices_by_id else ""
        dtname = f"{device}/{name}" if device else name
        self.builders[tid] = _TrackBuilder(
            name=name,
            device=device,
            track_type=track_type,
            fmt=fmt,
            unit=unit,
            srate=numbers["srate"],
            gain=numbers["gain"],
            offset=numbers["offset"],
            montype=montype,
            mindisp=numbers["mindisp"],
            maxdisp=numbers["maxdisp"],
            skip=self.track_names is not None and dtname not in self.track_names,
        )

    def _device_info(self, body: int, end: int) -> None:
        """Declare one device."""
        did = _U32(self.data, body)[0]
        type_name, p = _read_str(self.view, body + 4, end)
        name, p = _read_str(self.view, p, end)
        port = ""
        if p + 4 <= end:
            port, p = _read_str(self.view, p, end)
        self.devices_by_id[did] = VitalDevice(name=name or type_name, type_name=type_name, port=port)

    def _command(self, body: int) -> None:
        """Apply a display-order or event-reset command."""
        command = self.data[body]
        if command == _COMMAND_TRACK_ORDER:
            count = _U16(self.data, body + 1)[0]
            self.order_ids = list(np.frombuffer(self.data, dtype="<u2", count=count, offset=body + 3))
        elif command == _COMMAND_RESET_EVENTS:
            for builder in self.builders.values():
                if builder.name == "EVENT" and not builder.device:
                    builder.times.clear()
                    builder.sizes.clear()
                    builder.strings.clear()


def read_vital(path: str | Path, *, track_names: set[str] | None = None) -> VitalCase:
    """Read one ``.vital`` file at native resolution.

    Args:
        path: The file.
        track_names: If given, keep only these ``device/track`` names. Every track is still declared in
            the file, but the records of the others are skipped.

    Returns:
        The case, with every kept track's records in file order.

    Raises:
        VitalFormatError: If the file is not a ``.vital`` file, a packet runs past the end of the file,
            or a record names a track that was never declared.
    """
    path = Path(path)
    try:
        with gzip.open(path, "rb") as fh:
            data = fh.read()
    except (OSError, EOFError) as error:
        raise VitalFormatError(f"{path}: not a readable gzip stream ({error})") from error
    header = _read_header(path, data)
    reader = _PacketReader(path, data, track_names)
    reader.read(10 + header.length)

    tracks: dict[str, VitalTrack] = {}
    seen: dict[str, int] = {}
    for builder in reader.builders.values():
        # A few files declare one device/track name under two track ids (case 2645 has two
        # Solar8000/BT streams, overlapping in time). The vitaldb library merges them into one track;
        # here they stay apart, the later one named "BT#2" (then "#3", ...), so no record is reinterpreted.
        base = f"{builder.device}/{builder.name}".lstrip("/")
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            builder.name = f"{builder.name}#{seen[base]}"
        if builder.skip:
            continue
        track = builder.build(header.packed)
        if track.dtname in tracks:
            raise VitalFormatError(f"{path}: track {track.dtname!r} is declared twice")
        tracks[track.dtname] = track
    names_by_id = {tid: f"{b.device}/{b.name}".lstrip("/") for tid, b in reader.builders.items()}
    display_order = [names_by_id[t] for t in reader.order_ids if t in names_by_id and names_by_id[t] in tracks]
    return VitalCase(
        path=path,
        version=header.version,
        dgmt=header.dgmt,
        header_start=header.start,
        header_end=header.end,
        packed=header.packed,
        devices={d.name: d for d in reader.devices_by_id.values()},
        tracks=tracks,
        display_order=display_order,
        unknown_packets=reader.unknown,
    )
