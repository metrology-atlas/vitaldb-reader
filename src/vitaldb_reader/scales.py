"""Corrections to stored scale factors that the release gets wrong.

A .vital track stores ``physical = offset + value x gain``. For a few track and factor combinations the stored
factors are wrong; each correction below was established in the VitalDB metrology atlas against an independent
value from the same device, across the whole release (19 October 2026):

- Primus CO2 stored at gain 0.1: the Primus's own ETCO2 numeric is 0.748 x the waveform peak in all 5,392 such
  recordings, and 0.998 after rescaling to 0.075 (the gain the other 950 recordings carry). Atlas observation
  primus-waveform-gains.
- Primus AWP stored at 0.1064 / -22 (910 recordings) or 0.01 / 0 (24): PIP is 0.976 or 4.49 x the waveform peak,
  0.995 after rescaling to 0.1 / -20 (the factors of the other 5,077 recordings). Same observation.
- SNUADC ART stored with the PLETH factors 0.394983 / -198.2504 (3 recordings): an arterial channel labelled
  with the plethysmogram's factors; the ART factors 0.987457 / -495.626 apply. Atlas observation
  art-with-pleth-factors.

Factors are matched to 6 significant figures, because the release stores the same factor at float or double
precision in different files.
"""

from __future__ import annotations

SCALE_RULES_VERSION = "2026-10-19"

# (device/track, stored gain, stored offset) -> (gain, offset, atlas observation)
CORRECTIONS: dict[tuple[str, float, float], tuple[float, float, str]] = {
    ("Primus/CO2", 0.1, 0.0): (0.075, 0.0, "primus-waveform-gains"),
    ("Primus/AWP", 0.1064, -22.0): (0.1, -20.0, "primus-waveform-gains"),
    ("Primus/AWP", 0.01, 0.0): (0.1, -20.0, "primus-waveform-gains"),
    ("SNUADC/ART", 0.394983, -198.2504): (0.987457, -495.626, "art-with-pleth-factors"),
}


def _sig(x: float) -> float:
    return float(f"{x:.6g}")


def corrected_scale(track: str, gain: float, offset: float) -> tuple[float, float, str | None]:
    """The scale factors to use for a track, and the reason when they differ from the stored ones.

    Args:
        track: ``device/track`` as the file names it (a re-declared track's ``#2`` suffix is ignored).
        gain: Stored gain.
        offset: Stored offset.

    Returns:
        ``(gain, offset, reason)``: the stored factors and ``None`` when no correction applies, otherwise the
        corrected factors and the atlas observation that establishes them.
    """
    name = track.split("#")[0]
    for (t, g, o), (cg, co, why) in CORRECTIONS.items():
        if t == name and _sig(gain) == _sig(g) and _sig(offset) == _sig(o):
            return cg, co, why
    return gain, offset, None
