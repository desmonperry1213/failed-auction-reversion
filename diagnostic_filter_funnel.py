# ═══════════════════════════════════════════════════════════════════════════════
# FILTER FUNNEL DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
#
# Investigates why trade counts are unexpectedly low across all four
# instruments in Phase 5. Runs the state machine TWICE on ES:
#
#     Run A : pre-registered behavior (gate_enabled=True)
#     Run B : daily VA overlap gate disabled (gate_enabled=False)
#
# Counter increments at every state transition expose where setups die.
# Side-by-side comparison of (A) and (B) reveals whether the gate is the
# choke point or whether the bottleneck is elsewhere (filters too strict,
# or some other logic issue).
#
# This is EXPLORATORY analysis. It does not modify pre-registered parameters.
# It simply runs an alternative configuration to inform a possible
# methodological revision. If the diagnostic confirms the gate is the
# choke point, the proper next step is to formally re-pre-register the
# revised design (with explicit acknowledgment) and re-run cleanly.
#
# Outputs (saved to results\diagnostic\):
#     funnel_report.txt         — terminal-style funnel + comparison
#     trades_gate_enabled.csv   — Run A trades
#     trades_gate_disabled.csv  — Run B trades
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd

from strategy import Config, load_and_prepare_data
from tpo_state_machine import TPOStateMachine, trades_to_dataframe
from phase5_bootstrap import bootstrap_avg_r


OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\diagnostic"


def funnel_report(label, sm, total_bars):
    """Build a printable filter-funnel report for one state-machine run."""
    c = sm.counters
    lines = []
    lines.append("="*70)
    lines.append(f"  FILTER FUNNEL — {label}")
    lines.append("="*70)
    lines.append(f"  Total 1H bars processed       : {total_bars:>7,}")
    lines.append(f"  Closes outside weekly VA      : {c['closes_outside_va']:>7,}"
                 f"   ← state 1→2 entry")
    lines.append(f"    ↳ closed back inside w/o conf : {c['closed_back_no_conf']:>7,}")
    lines.append(f"    ↳ delta divergences detected  : {c['delta_divergences']:>7,}")
    lines.append(f"        ↳ failed min-distance     : {c['failed_min_distance']:>7,}")
    lines.append(f"        ↳ passed min-distance     : {c['passed_min_distance']:>7,}")
    lines.append(f"            ↳ failed velocity     : {c['failed_velocity']:>7,}")
    lines.append(f"            ↳ passed velocity     : {c['passed_velocity']:>7,}"
                 f"   ← state 2→3 (probe confirmed)")
    lines.append(f"  Probe confirmations           : {c['probe_confirmed']:>7,}")
    lines.append(f"  Back-inside events            : {c['back_inside_events']:>7,}"
                 f"   ← state 3 trigger")
    lines.append(f"    ↳ daily VA gate PASSED        : {c['gate_passed']:>7,}")
    lines.append(f"    ↳ daily VA gate FAILED        : {c['gate_failed']:>7,}"
                 f"   ← (filtered if gate enabled)")
    lines.append(f"  Retest waits started          : {c['retest_started']:>7,}"
                 f"   ← state 3→4")
    lines.append(f"    ↳ POC cancellations           : {c['poc_cancellations']:>7,}")
    lines.append(f"    ↳ retest filled               : {c['retest_filled']:>7,}"
                 f"   ← state 4→5 (entry)")
    lines.append(f"  VA acceptance resets          : {c['va_acceptance_resets']:>7,}")
    lines.append(f"  Trades closed                 : {c['trades_closed']:>7,}")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*72}")
    print("  FILTER FUNNEL DIAGNOSTIC")
    print(f"{'='*72}\n")

    # Force OHLC decomposition (matches pre-registered methodology)
    saved_delta = getattr(Config, 'DELTA_METHOD', 'close_vs_open')
    Config.DELTA_METHOD = 'ohlc_decomposition'

    # Use full 2015–2026 range, matching Phase 4
    saved_range = (Config.BACKTEST_START, Config.BACKTEST_END)
    Config.BACKTEST_START = "2015-01-01"
    Config.BACKTEST_END   = "2026-12-31"

    try:
        df_1h, df_1m = load_and_prepare_data()
    finally:
        Config.BACKTEST_START, Config.BACKTEST_END = saved_range

    total_bars = len(df_1h)
    print(f"\n  Dataset: {total_bars:,} 1H bars, {len(df_1m):,} 1M bars\n")

    # ── Run A: pre-registered (gate enabled) ─────────────────────────────────
    print("  Run A — daily VA gate ENABLED (pre-registered)...")
    t0 = time.time()
    smA = TPOStateMachine(df_1h, df_1m, contracts=1, verbose=False, gate_enabled=True)
    tradesA = smA.run()
    print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(tradesA):,}\n")

    # ── Run B: gate disabled ─────────────────────────────────────────────────
    print("  Run B — daily VA gate DISABLED (diagnostic)...")
    t0 = time.time()
    smB = TPOStateMachine(df_1h, df_1m, contracts=1, verbose=False, gate_enabled=False)
    tradesB = smB.run()
    print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(tradesB):,}\n")

    # Restore Config
    Config.DELTA_METHOD = saved_delta

    # ── Funnel reports ────────────────────────────────────────────────────────
    rep_A = funnel_report("RUN A — gate ENABLED (pre-registered)", smA, total_bars)
    rep_B = funnel_report("RUN B — gate DISABLED",                  smB, total_bars)
    print(rep_A)
    print()
    print(rep_B)

    # ── Side-by-side comparison ──────────────────────────────────────────────
    cmp_lines = []
    cmp_lines.append("\n" + "="*70)
    cmp_lines.append("  SIDE-BY-SIDE COMPARISON")
    cmp_lines.append("="*70)
    cmp_lines.append(f"  {'Counter':<32}{'Run A':>12}{'Run B':>12}{'Δ':>10}")
    cmp_lines.append("  " + "-"*66)
    keys_in_order = [
        'closes_outside_va', 'closed_back_no_conf', 'delta_divergences',
        'failed_min_distance', 'passed_min_distance',
        'failed_velocity', 'passed_velocity', 'probe_confirmed',
        'back_inside_events', 'gate_passed', 'gate_failed',
        'retest_started', 'poc_cancellations', 'retest_filled',
        'va_acceptance_resets', 'trades_closed',
    ]
    for k in keys_in_order:
        a = smA.counters[k]
        b = smB.counters[k]
        d = b - a
        cmp_lines.append(f"  {k:<32}{a:>12,}{b:>12,}{d:>+10,}")
    cmp_str = "\n".join(cmp_lines)
    print(cmp_str)

    # ── Bootstrap on Run B for performance check ─────────────────────────────
    print(f"\n{'='*70}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    dfA = trades_to_dataframe(tradesA)
    dfB = trades_to_dataframe(tradesB)

    perf_lines = []
    perf_lines.append("="*70)
    perf_lines.append("  PERFORMANCE COMPARISON")
    perf_lines.append("="*70)
    for label, dfX in [("Run A — gate enabled",  dfA),
                       ("Run B — gate disabled", dfB)]:
        if len(dfX) == 0:
            perf_lines.append(f"\n  {label}: 0 trades")
            continue
        rs   = dfX['r_multiple'].values
        boot = bootstrap_avg_r(rs)
        win  = (rs > 0).sum()
        wr   = win / len(dfX) * 100
        sumR = float(np.sum(rs))
        perf_lines.append(f"\n  {label}:")
        perf_lines.append(f"    n trades         : {len(dfX):,}")
        perf_lines.append(f"    Avg-R            : {boot['obs']:+.4f}")
        perf_lines.append(f"    95% CI           : [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
        perf_lines.append(f"    Bootstrap p      : {boot['p_value']:.4f}"
                          + (" *" if boot['p_value'] < 0.05 else ""))
        perf_lines.append(f"    Win rate         : {wr:.1f}%")
        perf_lines.append(f"    Sum of R         : {sumR:+.2f}")
    perf_str = "\n".join(perf_lines)
    print(perf_str)

    # ── Save reports + trades ─────────────────────────────────────────────────
    full = "\n".join([rep_A, "", rep_B, "", cmp_str, "", perf_str])
    out_summary = os.path.join(OUT_DIR, "funnel_report.txt")
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"\n  Saved: {out_summary}")

    dfA.to_csv(os.path.join(OUT_DIR, 'trades_gate_enabled.csv'),  index=False)
    dfB.to_csv(os.path.join(OUT_DIR, 'trades_gate_disabled.csv'), index=False)
    print(f"  Saved: trades_gate_enabled.csv  and  trades_gate_disabled.csv")

    # ── Interpretation hint ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  INTERPRETATION")
    print(f"{'='*70}")
    n_filtered_by_gate = smA.counters['gate_failed']
    n_setups_total     = smA.counters['back_inside_events']
    if n_setups_total > 0:
        gate_kill_rate = n_filtered_by_gate / n_setups_total * 100
        print(f"  The daily VA gate filtered {n_filtered_by_gate} of "
              f"{n_setups_total} setups ({gate_kill_rate:.1f}%).")
    if len(tradesB) > 0 and len(tradesA) > 0:
        ratio = len(tradesB) / max(len(tradesA), 1)
        print(f"  Removing the gate produced {ratio:.1f}× more trades.")
    print()
    print("  Read the comparison this way:")
    print("  • If Run B's avg-R is similar to or better than Run A's, the gate is")
    print("    not adding edge — it's filtering setups that perform comparably.")
    print("  • If Run B's avg-R is materially worse than Run A's, the gate is")
    print("    contributing real selectivity even at the cost of trade count.")
    print("  • The right next step depends on this comparison.")
    print()


if __name__ == "__main__":
    main()
