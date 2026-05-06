# ═══════════════════════════════════════════════════════════════════════════════
# NO-RETEST ENTRY-MODE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase 5 follow-up. The filter-funnel diagnostic showed that the daily VA
# gate IS providing real selectivity (avg-R drops from +0.095 to +0.010 when
# disabled), but ALSO showed that 72% of gate-passing setups die during the
# retest-waiting phase via POC cancellation. This is the larger choke point.
#
# This script tests an alternative entry mechanic: enter immediately at the
# close of the back-inside bar, skipping the retest requirement.
#
# Three runs on full ES 2015–2026:
#     A — gate=ON,  entry=retest        (pre-registered, baseline)
#     B — gate=ON,  entry=back_inside   (proposed fix: keep gate, drop retest)
#     C — gate=OFF, entry=back_inside   (most permissive: drop both filters)
#
# Side-by-side comparison reveals whether the retest mechanic is the actual
# bottleneck, and whether removing it preserves edge.
#
# Outputs (saved to results\diagnostic_noretest\):
#     funnel_report_noretest.txt
#     trades_A_baseline.csv
#     trades_B_no_retest.csv
#     trades_C_permissive.csv
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd

from strategy import Config, load_and_prepare_data
from tpo_state_machine import TPOStateMachine, trades_to_dataframe
from phase5_bootstrap import bootstrap_avg_r


OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\diagnostic_noretest"


def funnel_report(label, sm, total_bars):
    c = sm.counters
    lines = []
    lines.append("="*74)
    lines.append(f"  FILTER FUNNEL — {label}")
    lines.append("="*74)
    lines.append(f"  Total 1H bars processed         : {total_bars:>7,}")
    lines.append(f"  Closes outside weekly VA        : {c['closes_outside_va']:>7,}")
    lines.append(f"    ↳ closed back inside no conf  : {c['closed_back_no_conf']:>7,}")
    lines.append(f"    ↳ delta divergences           : {c['delta_divergences']:>7,}")
    lines.append(f"        ↳ failed min-distance     : {c['failed_min_distance']:>7,}")
    lines.append(f"        ↳ passed min-distance     : {c['passed_min_distance']:>7,}")
    lines.append(f"            ↳ failed velocity     : {c['failed_velocity']:>7,}")
    lines.append(f"            ↳ passed velocity     : {c['passed_velocity']:>7,}")
    lines.append(f"  Probe confirmations             : {c['probe_confirmed']:>7,}")
    lines.append(f"  Back-inside events              : {c['back_inside_events']:>7,}")
    lines.append(f"    ↳ daily VA gate PASSED        : {c['gate_passed']:>7,}")
    lines.append(f"    ↳ daily VA gate FAILED        : {c['gate_failed']:>7,}")
    lines.append(f"  Retest waits started            : {c['retest_started']:>7,}"
                 f"  ← only in 'retest' entry_mode")
    lines.append(f"    ↳ POC cancellations           : {c['poc_cancellations']:>7,}")
    lines.append(f"    ↳ retest filled               : {c['retest_filled']:>7,}")
    lines.append(f"  Immediate entries (back-inside) : {c['immediate_entries']:>7,}"
                 f"  ← only in 'back_inside' mode")
    lines.append(f"  VA acceptance resets            : {c['va_acceptance_resets']:>7,}")
    lines.append(f"  Trades closed                   : {c['trades_closed']:>7,}")
    return "\n".join(lines)


def perf_section(label, df_trades):
    lines = []
    lines.append(f"\n  {label}:")
    if len(df_trades) == 0:
        lines.append("    (no trades)")
        return "\n".join(lines)
    rs   = df_trades['r_multiple'].values
    boot = bootstrap_avg_r(rs)
    win  = (rs > 0).sum()
    wr   = win / len(df_trades) * 100
    sumR = float(np.sum(rs))
    avg_dol = float(df_trades['pnl_dollars'].mean())
    sum_dol = float(df_trades['pnl_dollars'].sum())
    lines.append(f"    n trades          : {len(df_trades):,}")
    lines.append(f"    Avg-R             : {boot['obs']:+.4f}")
    lines.append(f"    95% CI            : [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
    lines.append(f"    Bootstrap p       : {boot['p_value']:.4f}"
                 + (" *" if boot['p_value'] < 0.05 else ""))
    lines.append(f"    Win rate          : {wr:.1f}%")
    lines.append(f"    Sum of R          : {sumR:+.2f}")
    lines.append(f"    Avg trade $       : ${avg_dol:>10,.2f}")
    lines.append(f"    Sum $             : ${sum_dol:>10,.2f}")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*74}")
    print("  NO-RETEST ENTRY-MODE DIAGNOSTIC")
    print(f"{'='*74}\n")

    saved_delta = getattr(Config, 'DELTA_METHOD', 'close_vs_open')
    Config.DELTA_METHOD = 'ohlc_decomposition'

    saved_range = (Config.BACKTEST_START, Config.BACKTEST_END)
    Config.BACKTEST_START = "2015-01-01"
    Config.BACKTEST_END   = "2026-12-31"
    try:
        df_1h, df_1m = load_and_prepare_data()
    finally:
        Config.BACKTEST_START, Config.BACKTEST_END = saved_range

    total_bars = len(df_1h)
    print(f"\n  Dataset: {total_bars:,} 1H bars\n")

    # ── Run A: baseline (pre-registered) ─────────────────────────────────────
    print("  Run A — gate=ON, entry=retest (pre-registered baseline)...")
    t0 = time.time()
    smA = TPOStateMachine(df_1h, df_1m, contracts=1,
                          gate_enabled=True, entry_mode='retest')
    tradesA = smA.run()
    print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(tradesA):,}\n")

    # ── Run B: gate kept, retest dropped ─────────────────────────────────────
    print("  Run B — gate=ON, entry=back_inside (keep gate, drop retest)...")
    t0 = time.time()
    smB = TPOStateMachine(df_1h, df_1m, contracts=1,
                          gate_enabled=True, entry_mode='back_inside')
    tradesB = smB.run()
    print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(tradesB):,}\n")

    # ── Run C: most permissive ────────────────────────────────────────────────
    print("  Run C — gate=OFF, entry=back_inside (most permissive)...")
    t0 = time.time()
    smC = TPOStateMachine(df_1h, df_1m, contracts=1,
                          gate_enabled=False, entry_mode='back_inside')
    tradesC = smC.run()
    print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(tradesC):,}\n")

    Config.DELTA_METHOD = saved_delta

    # ── Funnels ───────────────────────────────────────────────────────────────
    rep_A = funnel_report("RUN A — gate=ON, entry=retest (pre-registered)",   smA, total_bars)
    rep_B = funnel_report("RUN B — gate=ON, entry=back_inside",                 smB, total_bars)
    rep_C = funnel_report("RUN C — gate=OFF, entry=back_inside (permissive)", smC, total_bars)
    print(rep_A); print(); print(rep_B); print(); print(rep_C)

    # ── Side-by-side comparison ──────────────────────────────────────────────
    cmp_lines = []
    cmp_lines.append("\n" + "="*74)
    cmp_lines.append("  SIDE-BY-SIDE COMPARISON")
    cmp_lines.append("="*74)
    cmp_lines.append(f"  {'Counter':<32}{'Run A':>10}{'Run B':>10}{'Run C':>10}")
    cmp_lines.append("  " + "-"*64)
    keys_in_order = [
        'closes_outside_va', 'delta_divergences', 'probe_confirmed',
        'back_inside_events', 'gate_passed', 'gate_failed',
        'retest_started', 'poc_cancellations', 'retest_filled',
        'immediate_entries', 'va_acceptance_resets', 'trades_closed',
    ]
    for k in keys_in_order:
        a = smA.counters[k]
        b = smB.counters[k]
        c = smC.counters[k]
        cmp_lines.append(f"  {k:<32}{a:>10,}{b:>10,}{c:>10,}")
    cmp_str = "\n".join(cmp_lines)
    print(cmp_str)

    # ── Performance ───────────────────────────────────────────────────────────
    perf_lines = []
    perf_lines.append("\n" + "="*74)
    perf_lines.append("  PERFORMANCE COMPARISON")
    perf_lines.append("="*74)
    dfA = trades_to_dataframe(tradesA)
    dfB = trades_to_dataframe(tradesB)
    dfC = trades_to_dataframe(tradesC)
    perf_lines.append(perf_section("Run A — gate=ON, entry=retest (pre-registered)", dfA))
    perf_lines.append(perf_section("Run B — gate=ON, entry=back_inside",               dfB))
    perf_lines.append(perf_section("Run C — gate=OFF, entry=back_inside",              dfC))
    perf_str = "\n".join(perf_lines)
    print(perf_str)

    # ── Save reports + trades ─────────────────────────────────────────────────
    full = "\n".join([rep_A, "", rep_B, "", rep_C, "", cmp_str, "", perf_str])
    out_summary = os.path.join(OUT_DIR, "funnel_report_noretest.txt")
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"\n  Saved: {out_summary}")
    dfA.to_csv(os.path.join(OUT_DIR, 'trades_A_baseline.csv'),    index=False)
    dfB.to_csv(os.path.join(OUT_DIR, 'trades_B_no_retest.csv'),   index=False)
    dfC.to_csv(os.path.join(OUT_DIR, 'trades_C_permissive.csv'),  index=False)
    print(f"  Saved: 3 trade CSVs")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("  INTERPRETATION")
    print(f"{'='*74}")
    print()
    print("  Compare Run A vs Run B (the key test):")
    print("    - Same gate (real selectivity preserved)")
    print("    - Different entry mechanic (retest vs immediate back-inside)")
    print()
    print("  Outcomes to look for:")
    print("    • B materially MORE trades and avg-R holds up well")
    print("        → retest was over-cautious; clean methodological case for")
    print("          re-registering the design with back_inside entry.")
    print("    • B materially MORE trades but avg-R collapses")
    print("        → retest WAS providing real edge through better entry pricing.")
    print("          The 72% POC-cancellation rate is a feature, not a bug —")
    print("          we were correctly avoiding setups that would have lost.")
    print("    • B about the same as A on both counts → no clear improvement,")
    print("          stick with pre-registered.")
    print()
    print("  Run C is the lower bound — what no filter looks like.")
    print()


if __name__ == "__main__":
    main()
