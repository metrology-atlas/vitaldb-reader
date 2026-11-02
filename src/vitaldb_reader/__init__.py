"""Read VitalDB ``.vital`` files at native resolution, with the Metrology Atlas's corrections.

The reader keeps every track at the resolution and in the dtype the file stores it, in file order, and resamples
nothing (:func:`read_vital`). The corrections the VitalDB metrology atlas found are separate, versioned steps:

- :mod:`vitaldb_reader.resets`: SNUADC records after a step-mode reset put back in sampling order
  (:data:`~vitaldb_reader.resets.RULE_VERSION`);
- :mod:`vitaldb_reader.scales`: corrected gain and offset of the tracks whose stored scale is wrong
  (:data:`~vitaldb_reader.scales.SCALE_RULES_VERSION`);
- :mod:`vitaldb_reader.timing_model`: the delay and window of each monitor numeric and the resolution of the
  clinical times (:data:`~vitaldb_reader.timing_model.MODEL_VERSION`);
- :mod:`vitaldb_reader.timing`, :mod:`vitaldb_reader.resolution`, :mod:`vitaldb_reader.lattice`: where samples
  sit in time, the value resolution of each track, and exact integer lattices of stored values.
"""

from vitaldb_reader.errors import VitalFormatError, VitalValidationError
from vitaldb_reader.resets import RULE_VERSION as RESET_RULE_VERSION
from vitaldb_reader.scales import SCALE_RULES_VERSION
from vitaldb_reader.timing_model import MODEL_VERSION
from vitaldb_reader.vital_file import (
    TRACK_NUMERIC,
    TRACK_STRING,
    TRACK_WAVE,
    VitalCase,
    VitalDevice,
    VitalTrack,
    read_vital,
)

__version__ = "0.1.0"

__all__ = [
    "MODEL_VERSION",
    "RESET_RULE_VERSION",
    "SCALE_RULES_VERSION",
    "TRACK_NUMERIC",
    "TRACK_STRING",
    "TRACK_WAVE",
    "VitalCase",
    "VitalDevice",
    "VitalFormatError",
    "VitalTrack",
    "VitalValidationError",
    "__version__",
    "read_vital",
]
