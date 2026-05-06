# Failed Auction Reversion in Liquid Futures Markets

Companion code for the working paper:

**"Pre-Registered Tests of Failed Weekly Auction Reversion in Liquid Futures Markets"**

*[Desmon Perry] — 2026*

Available at: [SSRN link — add when posted]

---

## Overview

This repository contains the complete code used to produce every result reported
in the paper. All findings are reproducible from the raw data using the scripts
provided. The pre-registration record documents the exact parameter values that
were locked before any confirmation backtest was run.

---

## Data

Data is **not included** in this repository. Purchase from
[Databento](https://databento.com):

| Product | Schema | Instruments | Date Range |
|---|---|---|---|
| CME Globex MDP 3.0 | OHLCV-1m | ES (continuous) | 2015-01-01 → 2026-05-06 |
| CME Globex MDP 3.0 | OHLCV-1m | NQ, GC, ZN (continuous) | 2015-01-01 → 2026-05-06 |

**Estimated cost: approximately $102 for all four instruments.**

Download as CSV, split by instrument. Place files in the following structure:

```
data/
  OHLCV-1M CME Globex Folder/
    OHLCV-1M CME Globex Data/
      OHLCV-1M CME Globex.csv                          ← ES data
  OHLCV-1M CME Globex Cross Instrument/
    OHLCV-1M CME Globex Cross Instrument Data/
      OHLCV-1M CME Globex Cross Intrument.csv          ← NQ, GC, ZN combined
```

> **Note:** The filename `Cross Intrument` contains a typo in the Databento
> delivery — use exactly as written above or the data loader will not find the file.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested on Python 3.12+. Compatible with Windows, macOS, and Linux.

---

## Reproducing the Paper Results

Run scripts in the order shown. Each script prints its results to the terminal
and saves outputs to `results/` subdirectories.

### Step 1 — Validate the TPO Engine (Section 3.3)

```bash
python test_tpo.py          # Visual validation — inspect charts against TradingView
python test_va_inline.py    # Automated unit tests — all should pass with no errors
```

`test_tpo.py` requires human inspection of the generated charts. Set `WEEKS_TO_VIEW`
in the script to any weeks you want to verify against TradingView's Fixed Range
Volume Profile indicator (Session Volume, US/Central timezone, 70% VA threshold).

### Step 2 — 1H Pre-Registration (Section 3.5, Figures 7 and 9)

```bash
python pre_registration_analysis.py
```

Produces probe distance/ATR and velocity score distributions from 30,487 probe
events. Outputs: `results/pre_registration/probe_distributions.png`,
`stability_check.png`, `probe_distribution.csv`.

Locked values confirmed: `SL_ATR_MULT_TPO=1.578`, `MIN_PROBE_DIST_ATR=0.40`,
`VELOCITY_THRESHOLD=2.21`.

### Step 3 — 1H Confirmation Backtest (Section 5.1, Table 1, Figure 1)

```bash
python run_tpo_confirmation.py
```

Runs the pre-registered state machine on full ES 2015–2026 at 1H resolution.
Produces 100 trades. Outputs: `results/confirmation_backtest/trades.csv`,
`summary.txt`, `equity_curve.png`.

Expected: avg-R = +0.0949, bootstrap p = 0.1945, n = 100.

### Step 4 — Bootstrap and Sub-Period Analysis (Section 5.1)

```bash
python phase5_bootstrap.py
```

Runs 10,000-resample bootstrap tests including sub-period and ATR-tertile
conditional sub-tests. Fixed seed: 42.

### Step 5 — 1H Cross-Instrument Replication (Section 5.2, Table 2, Figure 3)

```bash
python phase5b_cross_instrument.py
```

Runs identical 1H design on NQ, GC, ZN. Computes Fisher's combined test.
Expected: Fisher combined p = 0.2186.

### Step 6 — 5M Pre-Registration (Section 3.5, Figures 8 and 9)

```bash
python pre_registration_analysis_5m.py
```

Produces probe distributions from 361,496 probe events at 5M resolution.
Runtime: approximately 5 minutes.

Locked values confirmed: `SL_ATR_MULT_5M=4.544`, `MIN_PROBE_DIST_ATR_5M=1.045`,
`VELOCITY_THRESHOLD_5M=3.808`.

### Step 7 — 5M Confirmation Backtest (Section 5.3, Table 3, Figure 2)

```bash
python run_tpo_confirmation_5m.py
```

Runs the re-pre-registered state machine on full ES 2015–2026 at 5M resolution.
Runtime: approximately 5–10 minutes. Produces 1,017 trades.
Expected: avg-R = +0.0595, bootstrap p = 0.0141, n = 1,017.

### Step 8 — 5M Cross-Instrument Replication (Section 5.4, Table 4, Figure 4)

```bash
python phase5b_cross_instrument_5m.py
```

Runs identical 5M design on NQ, GC, ZN. Computes Fisher's combined test.
Runtime: approximately 25–35 minutes (4 instruments × 800k bars each).
Expected: Fisher combined p = 0.0910.

### Step 9 — Diagnostic Chain (Section 4, Figure 6)

```bash
python diagnostic_filter_funnel.py     # Section 4.2 — filter funnel
python diagnostic_no_retest.py         # Section 4.3 — gate and retest selectivity
python diagnostic_timeframes.py        # Section 4.4 — bar-resolution artifact
```

These diagnostics reproduce the empirical investigation documented in Section 4.
They are exploratory analysis — not pre-registered tests — and use configurations
that differ from the pre-registered design to characterise the mechanism.

### Step 10 — Delta Proxy Comparison (Section 5.5)

```bash
python delta_diagnostic.py
```

Produces the sign-discordance comparison between OHLC-decomposition and binary
close-vs-open delta classification on 37,399 hourly ES bars. Expected: 26.1%
sign-discordance concentrated in high-volume bars.

---

## Pre-Registration Record

See [`PRE_REGISTRATION_RECORD.md`](PRE_REGISTRATION_RECORD.md) for the complete
dated parameter locking record. Parameters were locked before any confirmation
backtest was run. They are not modified here.

Two pre-registrations are documented:
- **1H** — locked 2026-05-05
- **5M** — locked 2026-05-06 (following bar-resolution artifact diagnostic)

---

## Reproducibility

All bootstrap tests use random seed 42 for deterministic output. Given the same
Databento input data and Python environment, every result in the paper is
exactly reproducible.

The pre-registration record contains the exact parameter values used. Any
researcher who purchases the same data and runs the same scripts will produce
identical results.

---

## Repository Structure

```
tpo_engine.py                    TPO volume-at-price engine
tpo_state_machine.py             Six-state strategy state machine
pre_registration_analysis.py     1H parameter derivation (pre-registration)
pre_registration_analysis_5m.py  5M parameter derivation (re-pre-registration)
run_tpo_confirmation.py          1H Phase 4 confirmation backtest
run_tpo_confirmation_5m.py       5M Phase 4 confirmation backtest
phase5_bootstrap.py              Bootstrap testing harness
phase5b_cross_instrument.py      1H cross-instrument replication
phase5b_cross_instrument_5m.py   5M cross-instrument replication
diagnostic_filter_funnel.py      Filter funnel diagnostic (Section 4.2)
diagnostic_no_retest.py          Gate and retest selectivity (Section 4.3)
diagnostic_timeframes.py         Bar-resolution artifact (Section 4.4)
delta_diagnostic.py              Delta proxy comparison (Section 5.5)
test_va_inline.py                Automated VA algorithm unit tests
test_tpo.py                      Visual TPO engine validation tooling
PRE_REGISTRATION_RECORD.md       Dated parameter locking record (Appendix A)
requirements.txt                 Python dependencies
```

---

## License

MIT License. You are free to use, modify, and build on this code for any
purpose including commercial use, with attribution.

---

## Citation

If you use this code or methodology in your own research, please cite:

```
[Author Name]. "Pre-Registered Tests of Failed Weekly Auction Reversion
in Liquid Futures Markets." SSRN Working Paper, 2026.
Available at: [SSRN link]
```

---

## Note on Prior Work

The original Initial Balance strategy that preceded this research — a
pattern-recognition approach that motivated the theoretical pivot documented
in Section 1 of the paper — is not included in this repository as it does
not reproduce any result reported in the paper. It is available on request.

---

## Contact

[Your preferred contact — LinkedIn profile URL or email]
