# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests the same pre-registered design across multiple bar resolutions to
# investigate whether the 1H bar choice was collapsing intraday sequence
# (back-inside → retest → POC touch) into single bars where POC cancellation
# triggered before retest fill.
#
# Hypothesis: smaller timeframes preserve event ordering and should:
#   • reduce POC cancellations as a fraction of setups
#   • increase trade count
#   • potentially improve avg-R if the underlying mechanism is real
#
# IMPORTANT: This is EXPLORATORY analysis. The pre-registered parameters
# (SL_ATR_MULT, MIN_PROBE_DIST_ATR, VELOCITY_THRESHOLD) were derived from
# 1H probe distributions. Running them on alternative timeframes does NOT
# constitute a re-run of the pre-registered hypothesis test. ATR-normalization
# makes the parameters approximately scale-invariant, but the formal
# pre-registration stands at 1H.
#
# Timeframes tested:
#   5M, 15M, 30M, 1H (baseline), 2H, 4H
#
# 5M will be slow — comment it out if runtime is a problem.
#
# Outputs (saved to results\diagnostic_timeframes\):
#     timeframe_summary.txt
#     trades_TF{5M,15M,30M,1H,2H,4H}.csv
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd
import pytz

from strategy import Config, compute_delta_ohlc_decomposition
from tpo_state_machine import TPOStateMachine, trades_to_dataframe
from phase5_bootstrap import bootstrap_avg_r


OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\diagnostic_timeframes"


# ── Configuration: timeframes to test ────────────────────────────────────────
# Globex session is ~23 hours/day. bars_per_day = 23 × 60 / minutes_per_bar.
# Comment out '5M' to skip if runtime is a concern (it adds ~15-20 min).
TIMEFRAMES = [
    {'name': '5M',  'freq': '5min',  'bars_per_day': 276},
    {'name': '15M', 'freq': '15min', 'bars_per_day':  92},
    {'name': '30M', 'freq': '30min', 'bars_per_day':  46},
    {'name': '1H',  'freq': 'h',     'bars_per_day':  23},   # baseline
    {'name': '2H',  'freq': '2h',    'bars_per_day':  12},
    {'name': '4H',  'freq': '4h',    'bars_per_day':   6},
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_es_1m_data():
    """Load and prepare ES 1M data (front-month, US/Central, with delta)."""
    CT = pytz.timezone(Config.TIMEZONE)
    PATH_1M = (
        r"C:\Trading\InitialBalanceAuctionFade\data"
        r"\OHLCV-1M CME Globex Folder"
        r"\OHLCV-1M CME Globex Data"
        r"\OHLCV-1M CME Globex.csv"
    )
    print(f"Loading ES 1M data from {PATH_1M}...")
    t0 = time.time()
    df = pd.read_csv(PATH_1M, parse_dates=['ts_event'])
    print(f"  Raw 1M rows : {len(df):,}  ({time.time()-t0:.0f}s)")

    df = df[~df['symbol'].str.contains('-', na=False)].copy()
    df['ts_event'] = df['ts_event'].dt.tz_convert(CT)

    # Front-month
    df['date'] = df['ts_event'].dt.date
    daily_vol = (
        df.groupby(['date', 'symbol'])['volume']
        .sum().reset_index()
        .rename(columns={'volume': 'daily_vol'})
    )
    front_month = (
        daily_vol.sort_values('daily_vol', ascending=False)
        .groupby('date').first().reset_index()
        [['date', 'symbol']].rename(columns={'symbol': 'front_symbol'})
    )
    df = df.merge(front_month, on='date', how='left')
    df = df[df['symbol'] == df['front_symbol']].copy()
    df = df.drop(columns=['date', 'front_symbol'])
    df = df.sort_values('ts_event').reset_index(drop=True)
    print(f"  Front-month : {len(df):,}")

    # OHLC-decomposition delta on 1M
    print(f"  Computing OHLC-decomposition delta on 1M bars...")
    up, down = compute_delta_ohlc_decomposition(df)
    df['delta_1m'] = up - down

    # Filter to backtest range
    start = pd.Timestamp("2015-01-01", tz=CT)
    end   = pd.Timestamp("2026-12-31", tz=CT)
    df    = df[(df['ts_event'] >= start) & (df['ts_event'] < end)].reset_index(drop=True)

    # Standardize column name to 'datetime' for state machine
    df = df.rename(columns={'ts_event': 'datetime'})
    print(f"  In range    : {len(df):,}")
    print(f"  Total elapsed: {time.time()-t0:.0f}s\n")
    return df


def aggregate_to_timeframe(df_1m, freq):
    """Aggregate 1M bars to a custom timeframe with summed delta."""
    df = df_1m.copy()
    df['bucket'] = df['datetime'].dt.floor(freq)
    bars = df.groupby('bucket').agg(
        open   = ('open',     'first'),
        high   = ('high',     'max'),
        low    = ('low',      'min'),
        close  = ('close',    'last'),
        volume = ('volume',   'sum'),
        delta  = ('delta_1m', 'sum'),
    ).reset_index().rename(columns={'bucket': 'datetime'})
    return bars


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def run_timeframe(df_1m, tf_cfg):
    """Aggregate to the timeframe and run the pre-registered state machine."""
    print(f"\n  ─── Timeframe: {tf_cfg['name']} ({tf_cfg['freq']}) ───")
    df_strategy = aggregate_to_timeframe(df_1m, tf_cfg['freq'])
    print(f"    Strategy bars : {len(df_strategy):,}")
    print(f"    Bars/day      : {tf_cfg['bars_per_day']}")

    t0 = time.time()
    sm = TPOStateMachine(
        df_1h        = df_strategy,
        df_1m        = df_1m,
        contracts    = 1,
        verbose      = False,
        gate_enabled = True,
        entry_mode   = 'retest',
        bar_freq     = tf_cfg['freq'],
        bars_per_day = tf_cfg['bars_per_day'],
    )
    trades = sm.run()
    runtime = time.time() - t0
    print(f"    Done in {runtime:.0f}s.  Trades: {len(trades):,}")

    return sm, trades, len(df_strategy)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_summary_row(tf_name, n_bars, sm, dfX):
    c = sm.counters
    if len(dfX) == 0:
        return (f"  {tf_name:<6}{n_bars:>9,}{c['back_inside_events']:>9,}"
                f"{c['gate_passed']:>8,}{c['retest_started']:>9,}"
                f"{c['poc_cancellations']:>8,}{c['retest_filled']:>8,}"
                f"  {'(no trades)':>40}")
    rs   = dfX['r_multiple'].values
    boot = bootstrap_avg_r(rs)
    win  = (rs > 0).sum() / len(dfX) * 100
    sumR = float(np.sum(rs))
    sig  = "*" if boot['p_value'] < 0.05 else " "
    return (f"  {tf_name:<6}{n_bars:>9,}{c['back_inside_events']:>9,}"
            f"{c['gate_passed']:>8,}{c['retest_started']:>9,}"
            f"{c['poc_cancellations']:>8,}{c['retest_filled']:>8,}"
            f"  {boot['obs']:>+8.4f}{sumR:>+9.2f}{win:>7.1f}%"
            f"{boot['p_value']:>9.4f}{sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*92}")
    print("  MULTI-TIMEFRAME DIAGNOSTIC")
    print(f"{'='*92}\n")
    print("  Hypothesis: smaller timeframes preserve event ordering")
    print("              (back-inside → retest → POC touch),")
    print("              reducing the bar-collapse artifact that triggers")
    print("              POC cancellation before retest fill on 1H.\n")
    print("  IMPORTANT: This is EXPLORATORY. Pre-registered parameters")
    print("             were locked from 1H probe distributions. Running")
    print("             them on other timeframes is informative but does")
    print("             not constitute a clean re-run of the pre-registered")
    print("             hypothesis test.\n")

    df_1m = load_es_1m_data()

    results = []  # list of (tf_cfg, sm, df_trades, n_bars)
    for tf in TIMEFRAMES:
        sm, trades, n_bars = run_timeframe(df_1m, tf)
        df_trades = trades_to_dataframe(trades)
        df_trades.to_csv(os.path.join(OUT_DIR, f"trades_TF{tf['name']}.csv"), index=False)
        results.append((tf, sm, df_trades, n_bars))

    # ── Summary table ─────────────────────────────────────────────────────────
    lines = []
    lines.append("\n" + "="*92)
    lines.append("  SUMMARY ACROSS TIMEFRAMES")
    lines.append("="*92)
    lines.append("")
    lines.append("                                                            "
                 "        Performance metrics")
    lines.append(f"  {'TF':<6}{'bars':>9}{'B_in':>9}{'gate=':>8}"
                 f"{'retest=':>9}{'poc_x':>8}{'fills':>8}"
                 f"  {'avg-R':>8}{'sum-R':>9}{'win':>8}{'boot_p':>10}")
    lines.append("  " + "-"*88)

    for tf, sm, dfX, n_bars in results:
        lines.append(fmt_summary_row(tf['name'], n_bars, sm, dfX))

    lines.append("")
    lines.append("  Column legend:")
    lines.append("    bars      total strategy-timeframe bars")
    lines.append("    B_in      back-inside events (state 3 trigger)")
    lines.append("    gate=     setups where daily VA gate PASSED")
    lines.append("    retest=   setups that entered State 4 retest waiting")
    lines.append("    poc_x     POC cancellations during retest")
    lines.append("    fills     retest filled (entered trade)")
    lines.append("    avg-R     primary test statistic")
    lines.append("    sum-R     cumulative R")
    lines.append("    win       win rate %")
    lines.append("    boot_p    bootstrap p-value (one-tailed, * if < 0.05)")

    # ── Cancellation rate analysis ───────────────────────────────────────────
    lines.append("")
    lines.append("="*92)
    lines.append("  POC CANCELLATION RATE BY TIMEFRAME")
    lines.append("="*92)
    lines.append("  Hypothesis check: does the cancellation rate fall as")
    lines.append("  timeframe shrinks (i.e., as bar collapse is reduced)?\n")
    lines.append(f"  {'TF':<6}{'retest_started':>16}{'poc_cancel':>14}{'cancel_rate':>14}")
    lines.append("  " + "-"*52)
    for tf, sm, _, _ in results:
        rs = sm.counters['retest_started']
        pc = sm.counters['poc_cancellations']
        rate = pc / rs * 100 if rs > 0 else 0.0
        lines.append(f"  {tf['name']:<6}{rs:>16,}{pc:>14,}{rate:>13.1f}%")

    summary = "\n".join(lines)
    print(summary)

    # Save
    out_path = os.path.join(OUT_DIR, "timeframe_summary.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n  Saved: {out_path}\n")

    # ── Interpretation hint ──────────────────────────────────────────────────
    print("="*92)
    print("  INTERPRETATION")
    print("="*92)
    print()
    print("  Compare 1H baseline (Run A reference: 100 trades, +0.0949 avg-R)")
    print("  to other timeframes:")
    print()
    print("  • If POC cancellation rate FALLS on smaller timeframes AND avg-R")
    print("    HOLDS or IMPROVES → bar resolution was the artifact. Strong")
    print("    case for re-pre-registering on a smaller timeframe.")
    print()
    print("  • If cancellation rate falls but avg-R also falls → smaller bars")
    print("    just admit more trades, mostly losers. Pre-registered TF was")
    print("    correctly calibrated.")
    print()
    print("  • If patterns are mostly flat across TFs → bar resolution is not")
    print("    a meaningful factor for this strategy.")
    print()


if __name__ == "__main__":
    main()
