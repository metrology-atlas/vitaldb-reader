"""Tests of the stored-scale corrections."""

import pytest

from vitaldb_reader.scales import corrected_scale


@pytest.mark.parametrize(
    ("track", "gain", "offset", "expected"),
    [
        ("Primus/CO2", 0.1, 0.0, (0.075, 0.0, "primus-waveform-gains")),
        ("Primus/CO2", 0.075, 0.0, (0.075, 0.0, None)),
        ("Primus/AWP", 0.1064, -22.0, (0.1, -20.0, "primus-waveform-gains")),
        ("Primus/AWP", 0.01, 0.0, (0.1, -20.0, "primus-waveform-gains")),
        ("Primus/AWP", 0.1, -20.0, (0.1, -20.0, None)),
        ("SNUADC/ART", 0.3949828125, -198.2504, (0.987457, -495.626, "art-with-pleth-factors")),
        ("SNUADC/PLETH", 0.3949828125, -198.2504, (0.3949828125, -198.2504, None)),
        ("SNUADC/ART#2", 0.394983, -198.2504, (0.987457, -495.626, "art-with-pleth-factors")),
    ],
)
def test_corrected_scale(track, gain, offset, expected):
    assert corrected_scale(track, gain, offset) == expected


