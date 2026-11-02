"""Write a small synthetic ``.vital`` release shaped like the real one. No real data is used.

The shape follows a structure-only inventory of the PhysioNet release (aggregates, no values):

- format version 3, a 10-byte header that states only the time-zone field, unpacked records;
- record time stamps near the year 2100, each waveform record stamped by adding its duration in
  float64, so 500 Hz records drift by 2**-20 relative, as in the release;
- SNUADC: uint16 ADC counts, 100-sample records at 500 Hz, gain/offset from the release;
- Primus: uint16 counts, 16-sample records at 62.5 Hz; BIS: float32 values on a 0.01 grid at 128 Hz;
- Solar8000 numerics on irregular ~2 s times, float32 on integer and 0.1 grids, with repeated stamps;
- an ``EVENT`` string track with no device, and a string track on a device;
- one gap, one overlap and one record that steps back in time.

Run ``python make_fixture.py OUT_DIR`` to regenerate ``fixtures/`` (deterministic).
"""

from __future__ import annotations

import gzip
from pathlib import Path
import struct
import sys

import numpy as np


T0 = 4102444800.0 + 12345.678  # 2100-01-01 plus an arbitrary offset, as the release does


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<L", len(raw)) + raw


def _packet(ptype: int, body: bytes) -> bytes:
    return struct.pack("<BL", ptype, len(body)) + body


def write_vital(path: Path) -> None:
    rng = np.random.default_rng(7)
    out = bytearray(b"VITA" + struct.pack("<L", 3) + struct.pack("<H", 10) + struct.pack("<h", 0) + bytes(8))
    devices = {1: ("SNUADC", "SNUADC"), 2: ("Solar 8000M", "Solar8000"), 3: ("Primus", "Primus"), 4: ("BIS", "BIS")}
    for did, (dtype_, name) in devices.items():
        out += _packet(9, struct.pack("<L", did) + _s(dtype_) + _s(name) + _s(f"COM{did}"))
    tracks = [  # tid, type, fmt, name, unit, srate, gain, offset, did
        (1, 1, 6, "ECG_II", "mV", 500.0, 0.0098745703125, -4.95626, 1),
        (2, 1, 6, "ART", "mmHg", 500.0, 0.98745703125, -495.626, 1),
        (3, 1, 6, "AWP", "hPa", 62.5, 0.1, -20.0, 3),
        (4, 1, 1, "EEG1_WAV", "uV", 128.0, 1.0, 0.0, 4),
        (5, 2, 1, "HR", "/min", 0.0, 1.0, 0.0, 2),
        (6, 2, 1, "BT", "℃", 0.0, 1.0, 0.0, 2),
        (7, 5, 0, "GAS2_AGENT", "", 0.0, 0.0, 0.0, 2),
        (8, 5, 0, "EVENT", "", 0.0, 0.0, 0.0, 0),
        (9, 2, 1, "BT", "℃", 0.0, 1.0, 0.0, 2),  # a second track under the same name, as case 2645 has
    ]
    for tid, ttype, fmt, name, unit, srate, gain, offset, did in tracks:
        body = (
            struct.pack("<HBB", tid, ttype, fmt)
            + _s(name)
            + _s(unit)
            + struct.pack("<ffL", 0.0, 100.0, 0)
            + struct.pack("<f", srate)
            + struct.pack("<dd", gain, offset)
            + struct.pack("<B", 0)
            + struct.pack("<L", did)
        )
        out += _packet(0, body)

    recs: list[tuple[float, int, bytes]] = []
    arrival: dict[int, float] = {}  # records written at a time other than their stamp

    def wave(  # noqa: PLR0913, PLR0917 - one argument per record-layout field
        tid: int, start: float, n_records: int, per: int, srate: float, make
    ) -> float:
        t = start
        for _ in range(n_records):
            recs.append((t, tid, struct.pack("<L", per) + make(per).tobytes()))
            t += per / srate  # accumulated in float64 at ~4e9 s: the release's drift
        return t

    def counts(lo: int, hi: int):
        return lambda n: rng.integers(lo, hi, n).astype("<u2")

    # ECG: 30 s, a 2.5 s gap, 20 s more, then one record that starts 40 samples early (overlap)
    t = wave(1, T0 + 1.25, 150, 100, 500.0, counts(450, 560))
    t = wave(1, t + 2.5, 100, 100, 500.0, counts(450, 560))
    wave(1, t - 40 / 500.0, 20, 100, 500.0, counts(450, 560))
    # ART: 50 s, with one record that steps back in time in the middle
    t = wave(2, T0 + 1.5, 120, 100, 500.0, counts(560, 640))
    # written when it arrived (after the record before it), but stamped a second earlier
    recs.append((t - 1.0, 2, struct.pack("<L", 100) + counts(560, 640)(100).tobytes()))
    arrival[len(recs) - 1] = t
    wave(2, t + 0.2, 120, 100, 500.0, counts(560, 640))
    wave(3, T0, 220, 16, 62.5, counts(200, 400))
    wave(4, T0 + 0.5, 400, 16, 128.0, lambda n: (rng.integers(-20000, 20000, n) * 0.01).astype("<f4"))
    for i in range(30):
        t = T0 + 2.0 + 2.0 * i + (0.0 if i % 7 else 0.013)
        recs.append((t, 5, struct.pack("<f", float(rng.integers(55, 95)))))
        if i % 10 == 3:  # a repeated time stamp with a different value, as the release has
            recs.append((t, 5, struct.pack("<f", float(rng.integers(55, 95)))))
        recs.append((t + 0.5, 6, struct.pack("<f", np.float32(np.float32(rng.integers(360, 372)) * np.float32(0.1)))))
    for i in range(5):  # the second BT stream, overlapping the first in time
        recs.append((T0 + 20.25 + 2.0 * i, 9, struct.pack("<f", np.float32(35.0 + 0.1 * i))))
    recs.append((T0 + 3.0, 7, struct.pack("<L", 0) + _s("SEVO")))
    for t, label in (
        (T0, "Case started"),
        (T0 + 10.0, "Surgery started"),
        (T0 + 55.0, "Surgery finished"),
        (T0 + 59.0, "Case finished"),
    ):
        recs.append((t, 8, struct.pack("<L", 0) + _s(label)))
    # the release interleaves tracks by time; keep each track's own order (including the step back)
    order = sorted(range(len(recs)), key=lambda i: (arrival.get(i, recs[i][0]), i))
    for i in order:
        t, tid, payload = recs[i]
        out += _packet(1, struct.pack("<HdH", 10, t, tid) + payload)
    out += _packet(6, struct.pack("<BH", 5, 8) + np.arange(1, 9, dtype="<u2").tobytes())
    path.write_bytes(gzip.compress(bytes(out), compresslevel=9, mtime=0))


CLINICAL_HEADER = (
    "caseid,subjectid,casestart,caseend,anestart,aneend,opstart,opend,adm,dis,icu_days,"
    "death_inhosp,age,sex,height,weight,bmi,asa,emop,department,optype,dx,opname,approach,"
    "position,ane_type,preop_hb,intraop_ebl"
)
CLINICAL_ROW = (
    "1,901,0,60,-120,58,10,55,-86400,400000,2,0,63,F,160.2,55.1,21.5,2,0,General surgery,Colorectal,"
    "Synthetic dx,Synthetic op,Open,Supine,General,12.9,300"
)


def main(out_dir: str) -> None:
    root = Path(out_dir)
    (root / "vital_files").mkdir(parents=True, exist_ok=True)
    write_vital(root / "vital_files" / "0001.vital")
    (root / "clinical_data.csv").write_text(CLINICAL_HEADER + "\n" + CLINICAL_ROW + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
