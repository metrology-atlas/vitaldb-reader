# vitaldb-reader

Reads VitalDB `.vital` files at native resolution, with the corrections found by the
[VitalDB metrology atlas](https://vitaldb.metrology-atlas.org). MIT licence; needs Python 3.11+ and numpy.

The reader keeps every track at the resolution and in the dtype the file stores it, in file order, and resamples
nothing: no shared grid, no filled gaps. Each correction the atlas found is a separate, versioned step, so an
analysis can say exactly which it applied.

## Install

```bash
pip install "vitaldb-reader @ git+https://github.com/metrology-atlas/vitaldb-reader@0.1.0"
```

## Use

```python
from vitaldb_reader import read_vital
from vitaldb_reader.resets import reorder_case

case = read_vital("vital_files/0001.vital", track_names={"SNUADC/ECG_II", "Solar8000/HR"})
case, orders = reorder_case(case)          # SNUADC records after each reset back in sampling order
ecg = case.tracks["SNUADC/ECG_II"]
values = ecg.physical()                    # value * gain + offset for integer formats
```

| Module | What it does | Version |
|---|---|---|
| `vital_file` | Reads the `.vital` container: devices, tracks, records, exact record times | — |
| `resets` | Puts the SNUADC records after a step-mode reset back in sampling order | `RULE_VERSION` |
| `scales` | Corrected gain and offset where the stored scale is wrong | `SCALE_RULES_VERSION` |
| `timing_model` | Delay and window of each monitor numeric; resolution of the clinical times | `MODEL_VERSION` |
| `timing` | Runs and segments of each waveform track; nominal rates | — |
| `resolution` | Value resolution of each track, converter limits, near-duplicate records | — |
| `lattice` | Exact integer lattices of stored values | — |

What each correction is and the evidence for it are on the atlas's pages; the scripts that measured them are in
[vitaldb-bench](https://github.com/metrology-atlas/vitaldb-bench).

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests use a small synthetic `.vital` file (`tests/fixtures`, made by `tests/make_fixture.py`); no real data.

## Credits

The `.vital` format is VitalDB's, documented and implemented in
[vitalutils](https://github.com/vitaldb/vitalutils) (BSD-3-Clause, the VitalDB authors). This reader is an
independent implementation and contains no code from it. It was first written as a dataset connector for
[TimeNet](https://github.com/OpenTSLM/TimeNet); the TimeNet adapter is offered there separately. Written with AI
assistance (Claude, Anthropic) and reviewed by the author.
