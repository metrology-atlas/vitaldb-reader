# Changelog

## 0.1.0 (2 November 2026)

First release as its own package. The code was written from October 2026 as a VitalDB connector inside a local
copy of TimeNet (`timenet_connectors.datasets.vitaldb`), which had no version control; this release starts its
history. Changes from that code: the package is `vitaldb_reader`; it needs numpy only (TimeNet's error classes
are replaced by `VitalFormatError` and `VitalValidationError`); the TimeNet adapter (`connector.py`) and its
tests are not included.

Corrections and their versions in this release:

- `timing_model.MODEL_VERSION = "vitaldb-timing-2026.11.1"`. Since 2026.10.3: Solar8000 HR is 60 over the mean of
  the last 8 R-R intervals within 15% of their median, truncated to 1 bpm (was: 60 over the median of the R-R
  intervals in the last 6 s); its window is 8 beats. Evidence: atlas study `hr-derivation`.
- `resets.RULE_VERSION = "snuadc-reset-order-2026.10.17"`: records after an SNUADC step-mode reset put back in
  sampling order. Evidence: atlas study `reset-cluster-order`.
- `scales.SCALE_RULES_VERSION = "2026-10-19"`: corrected scale factors. Evidence: atlas device and track pages.
