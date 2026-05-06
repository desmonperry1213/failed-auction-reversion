PRE-REGISTRATION RECORD — Initial Balance Auction Fade
Date locked: 2026-05-05
Dataset: ES front-month, 2015-01-01 to 2026-05-01, 67,151 1H bars
Probe events: 30,487

LOCKED PARAMETERS:
  SL_ATR_MULT          = 1.578    (median of distance/ATR distribution)
  MIN_PROBE_DIST_ATR   = 0.40     (10th percentile — heuristic, no natural break)
  VELOCITY_THRESHOLD   = 2.21     (90th percentile — start of spike-zone tail)

VOLUME NORMALIZATION:
  Production state machine uses rolling 60-day median hourly volume
  in the velocity score denominator. Threshold value is in normalized
  units and applies regardless of absolute volume regime.

STABILITY EVIDENCE:
  Distance/ATR median spread across 2015-17, 2018-20, 2021-23, 2024-26:
    0.036 (well below 0.10 "stable" threshold)
  Velocity median spread: 0.232 (above 0.20 threshold; attributed to
    global-median normalization artifact, addressed by rolling-median
    in production)

RECALIBRATION RULE:
  None. Parameters are locked indefinitely. Validity rests on
  ATR-normalization argument and the empirical stability evidence above.
  If post-deployment evidence emerges of structural break in distance/ATR
  distribution, recalibration would be performed via identical methodology
  on the most recent 60-month window, with full transparency.

THESE PARAMETERS DO NOT CHANGE BASED ON CONFIRMATION BACKTEST RESULTS.

---

## ADDENDUM — 5M RE-PRE-REGISTRATION
Date locked: 2026-05-06
Dataset: ES front-month, 2015-01-01 to 2026-05-01, 798,204 5M bars
Probe events: 361,496

LOCKED PARAMETERS (5M):
  SL_ATR_MULT_5M         = 4.544    (median of distance/ATR distribution — auto)
  MIN_PROBE_DIST_ATR_5M  = 1.045    (10th percentile — same methodology as 1H)
  VELOCITY_THRESHOLD_5M  = 3.808    (90th percentile — same methodology as 1H)

VOLUME NORMALIZATION:
  Rolling 60-day median × 276 bars/day = 16,560-bar window for 5M velocity scoring.

METHODOLOGICAL REASONING FOR RE-PRE-REGISTRATION:
  Original 1H pre-registration was tested via Phase 4 confirmation backtest
  (avg-R = +0.0949, p = 0.1945, null not rejected) and Phase 5b cross-instrument
  replication (Fisher combined p = 0.219, joint null not rejected).

  Diagnostic analysis (filter-funnel, no-retest, multi-timeframe) revealed that
  the 1H bar resolution was collapsing the back-inside → retest → POC-touch
  sequence into single bars where POC cancellation triggered before retest fill,
  killing 72% of post-gate setups. At 5M resolution this artifact is reduced
  (33% cancellation rate) and the per-trade edge resolves cleanly with sufficient
  sample size for statistical detection.

  The decision to re-pre-register at 5M was based on the methodological
  diagnostic chain, not on the 5M strategy result (which has not been computed
  under these new locked parameters at the time of locking).

STABILITY EVIDENCE (5M):
  Distance/ATR median spread across 2015-17, 2018-20, 2021-23, 2024-26: 0.602
  Velocity median spread: 0.055 (more stable than 1H's 0.232)
  Velocity 90th-pct spread: 2.254 (regime drift in tail — flagged for paper transparency)

  Distance drift is larger than 1H reference (0.036). Attributed to plausible
  microstructure shifts at the 5M scale (HFT growth, intraday range expansion).
  Reported transparently rather than corrected.

THESE PARAMETERS DO NOT CHANGE BASED ON CONFIRMATION BACKTEST RESULTS.
The original 1H pre-registration above this addendum remains valid as the
project's original committed test. The 5M locks are a methodologically motivated
second pre-registration; both will be reported in the paper.
