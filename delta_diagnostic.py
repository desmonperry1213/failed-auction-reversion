# ═══════════════════════════════════════════════════════════════════════════════
# DELTA METHOD DIAGNOSTIC — INITIAL BALANCE AUCTION FADE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Loads your 1M data once, computes BOTH delta methods (close-vs-open and
# OHLC-decomposition), aggregates to 1H bars, and shows you exactly where
# the two methods disagree and by how much.
#
# Use this BEFORE flipping Config.DELTA_METHOD in production. It tells you:
#   1. How correlated the two methods are (overall agreement)
#   2. How often they disagree on sign (most important for your strategy,
#      since your delta-mismatch rule depends on sign)
#   3. Which historical 1H bars they disagree on most extremely
#   4. Whether the April 27 missing bar resolves to the expected sign
#      under the new method
#
# Run from C:\Trading\InitialBalanceAuctionFade\:
#     python delta_diagnostic.py
#
# Outputs:
#   - terminal: summary stats, top divergent bars, target-week comparison
#   - C:\Trading\InitialBalanceAuctionFade\results\delta_comparison.csv
# ═══════════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import numpy as np
import pytz

import strategy
from strategy import (
    Config,
    compute_delta_close_vs_open,
    compute_delta_ohlc_decomposition,
)


# ── Configurable: which week to inspect in detail ──────────────────────────────
TARGET_WEEK_START = "2026-04-26"
TARGET_WEEK_END   = "2026-05-03"

# ── Configurable: how much history to scan ─────────────────────────────────────
SCAN_START = "2020-01-01"
SCAN_END   = "2026-12-31"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (computes BOTH deltas in one pass for efficiency)
# ═══════════════════════════════════════════════════════════════════════════════

def load_both_deltas():
    """
    Load 1H and 1M data, compute both delta methods on the 1M bars,
    aggregate to 1H, and return a single DataFrame with both delta columns.
    """
    CT = pytz.timezone(Config.TIMEZONE)

    PATH_1H = (
        r"C:\Trading\InitialBalanceAuctionFade\data"
        r"\OHLCV-1H CME Globex Folder"
        r"\OHLCV-1H CME Globex Data"
        r"\OHLCV-1H CME Globex.csv"
    )
    PATH_1M = (
        r"C:\Trading\InitialBalanceAuctionFade\data"
        r"\OHLCV-1M CME Globex Folder"
        r"\OHLCV-1M CME Globex Data"
        r"\OHLCV-1M CME Globex.csv"
    )

    print("Loading 1H data...")
    df_1h = pd.read_csv(PATH_1H, parse_dates=['ts_event'])
    print(f"  {len(df_1h):,} raw 1H rows loaded")

    print("Loading 1M data...")
    df_1m = pd.read_csv(PATH_1M, parse_dates=['ts_event'])
    print(f"  {len(df_1m):,} raw 1M rows loaded")

    df_1h = df_1h[~df_1h['symbol'].str.contains('-')].copy()
    df_1m = df_1m[~df_1m['symbol'].str.contains('-')].copy()

    df_1h['ts_event'] = df_1h['ts_event'].dt.tz_convert(CT)
    df_1m['ts_event'] = df_1m['ts_event'].dt.tz_convert(CT)

    # Build front-month continuous contract (same logic as strategy.py)
    def build_front_month(df):
        df['date'] = df['ts_event'].dt.date
        daily_vol  = (
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
        return df

    print("Building front-month continuous contract...")
    df_1h = build_front_month(df_1h)
    df_1m = build_front_month(df_1m)
    print(f"  1H front-month rows: {len(df_1h):,}")
    print(f"  1M front-month rows: {len(df_1m):,}")

    # ── Compute BOTH delta methods on the 1M bars ─────────────────────────────
    print("Computing both delta methods on 1M bars...")
    up_old, dn_old = compute_delta_close_vs_open(df_1m)
    up_new, dn_new = compute_delta_ohlc_decomposition(df_1m)
    df_1m['delta_old'] = up_old - dn_old
    df_1m['delta_new'] = up_new - dn_new
    df_1m['hour_bucket'] = df_1m['ts_event'].dt.floor('h')

    # Aggregate to 1H
    delta_1h = (
        df_1m.groupby('hour_bucket')
        .agg(delta_old=('delta_old', 'sum'),
             delta_new=('delta_new', 'sum'),
             total_vol=('volume',    'sum'))
        .reset_index()
    )

    # Merge onto 1H bars
    df_1h = df_1h.sort_values('ts_event').reset_index(drop=True)
    df_1h = df_1h.merge(
        delta_1h, left_on='ts_event', right_on='hour_bucket', how='left'
    )
    df_1h = df_1h.drop(columns=['hour_bucket'])
    df_1h = df_1h.rename(columns={'ts_event': 'datetime'})
    df_1h['delta_old'] = df_1h['delta_old'].fillna(0)
    df_1h['delta_new'] = df_1h['delta_new'].fillna(0)
    df_1h['total_vol'] = df_1h['total_vol'].fillna(0)

    # Filter to scan range
    start = pd.Timestamp(SCAN_START, tz=CT)
    end   = pd.Timestamp(SCAN_END,   tz=CT)
    df_1h = df_1h[(df_1h['datetime'] >= start) & (df_1h['datetime'] < end)].reset_index(drop=True)

    print(f"\nFinal dataframe: {len(df_1h):,} 1H bars")
    print(f"  From : {df_1h['datetime'].iloc[0]}")
    print(f"  To   : {df_1h['datetime'].iloc[-1]}")
    return df_1h


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_methods(df_1h):
    """Print aggregate statistics comparing the two delta methods."""
    d_old = df_1h['delta_old'].values
    d_new = df_1h['delta_new'].values

    # Drop bars with zero volume (no information either way)
    mask = df_1h['total_vol'] > 0
    d_old_m = d_old[mask]
    d_new_m = d_new[mask]

    print(f"\n{'═'*70}")
    print(f"  DELTA METHOD COMPARISON")
    print(f"{'═'*70}")
    print(f"  Bars with volume > 0 : {mask.sum():,} of {len(df_1h):,}")

    # Correlation
    if len(d_old_m) > 1:
        corr = np.corrcoef(d_old_m, d_new_m)[0, 1]
    else:
        corr = float('nan')

    print(f"\n  Pearson correlation  : {corr:+.4f}")
    print(f"    (1.0 = identical, 0.0 = unrelated, -1.0 = inverted)")

    # Sign-agreement (most important for your strategy)
    same_sign  = ((d_old_m > 0) & (d_new_m > 0)) | ((d_old_m < 0) & (d_new_m < 0))
    diff_sign  = ((d_old_m > 0) & (d_new_m < 0)) | ((d_old_m < 0) & (d_new_m > 0))
    one_zero   = (d_old_m == 0) | (d_new_m == 0)
    n          = len(d_old_m)

    print(f"\n  Sign agreement (both >0 or both <0)        : "
          f"{same_sign.sum():,}  ({same_sign.sum()/n*100:.1f}%)")
    print(f"  Sign disagreement (one >0 and other <0)    : "
          f"{diff_sign.sum():,}  ({diff_sign.sum()/n*100:.1f}%)")
    print(f"  At least one method = 0                    : "
          f"{one_zero.sum():,}  ({one_zero.sum()/n*100:.1f}%)")

    print(f"\n  >>> {diff_sign.sum()/n*100:.1f}% of 1H bars FLIP SIGN between methods")
    print(f"  This is the headline number: that fraction of your historical")
    print(f"  delta-mismatch decisions could change under the new method.")

    # Distribution stats
    print(f"\n  {'Metric':<28}{'close_vs_open':>16}{'ohlc_decomp':>16}")
    print(f"  {'-'*28}{'-'*16}{'-'*16}")
    print(f"  {'Mean':<28}{np.mean(d_old_m):>16,.0f}{np.mean(d_new_m):>16,.0f}")
    print(f"  {'Median':<28}{np.median(d_old_m):>16,.0f}{np.median(d_new_m):>16,.0f}")
    print(f"  {'Std':<28}{np.std(d_old_m):>16,.0f}{np.std(d_new_m):>16,.0f}")
    print(f"  {'Mean abs':<28}{np.mean(np.abs(d_old_m)):>16,.0f}{np.mean(np.abs(d_new_m)):>16,.0f}")
    print(f"  {'95th pct abs':<28}{np.percentile(np.abs(d_old_m), 95):>16,.0f}"
          f"{np.percentile(np.abs(d_new_m), 95):>16,.0f}")

    # Top divergent bars by absolute difference
    print(f"\n{'─'*70}")
    print(f"  TOP 20 BARS WHERE METHODS DISAGREE MOST")
    print(f"{'─'*70}")
    df_1h['delta_diff'] = df_1h['delta_new'] - df_1h['delta_old']
    df_1h['abs_diff']   = df_1h['delta_diff'].abs()
    top_div = df_1h.nlargest(20, 'abs_diff')[
        ['datetime', 'open', 'high', 'low', 'close',
         'total_vol', 'delta_old', 'delta_new', 'delta_diff']
    ].copy()

    # Pretty print
    print(f"\n{'Datetime':<25}{'Body':>10}{'Range':>10}"
          f"{'Vol':>12}{'Delta Old':>14}{'Delta New':>14}{'Δ Diff':>14}")
    print('─' * 99)
    for _, r in top_div.iterrows():
        body  = r['close'] - r['open']
        rng   = r['high'] - r['low']
        body_str = f"{body:+.2f}"
        print(f"{str(r['datetime'])[:19]:<25}"
              f"{body_str:>10}"
              f"{rng:>10.2f}"
              f"{r['total_vol']:>12,.0f}"
              f"{r['delta_old']:>+14,.0f}"
              f"{r['delta_new']:>+14,.0f}"
              f"{r['delta_diff']:>+14,.0f}")

    return df_1h


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET WEEK INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════

def show_target_week(df_1h, start, end):
    """Print bar-by-bar comparison for a specific date range."""
    CT = pytz.timezone(Config.TIMEZONE)
    s = pd.Timestamp(start, tz=CT)
    e = pd.Timestamp(end,   tz=CT)
    week = df_1h[(df_1h['datetime'] >= s) & (df_1h['datetime'] <= e)].copy()

    print(f"\n{'─'*70}")
    print(f"  TARGET WEEK: {start} to {end}")
    print(f"{'─'*70}")
    print(f"  {len(week)} 1H bars in window")

    # Find bars where the two methods disagree on sign
    disagree = week[
        ((week['delta_old'] > 0) & (week['delta_new'] < 0)) |
        ((week['delta_old'] < 0) & (week['delta_new'] > 0))
    ]
    print(f"  {len(disagree)} bars with sign disagreement\n")

    # Print all bars in the window in compact form
    print(f"{'Datetime':<22}{'O':>9}{'H':>9}{'L':>9}{'C':>9}"
          f"{'Body':>8}{'Vol':>10}{'Δ_old':>11}{'Δ_new':>11}{'Flag':>8}")
    print('─' * 116)
    for _, r in week.iterrows():
        body = r['close'] - r['open']
        flag = ''
        if (r['delta_old'] > 0 and r['delta_new'] < 0) or \
           (r['delta_old'] < 0 and r['delta_new'] > 0):
            flag = '*FLIP*'
        elif r['delta_old'] == 0 and r['delta_new'] != 0:
            flag = '!new'
        elif r['delta_old'] != 0 and r['delta_new'] == 0:
            flag = '!old'

        print(f"{str(r['datetime'])[:19]:<22}"
              f"{r['open']:>9.2f}"
              f"{r['high']:>9.2f}"
              f"{r['low']:>9.2f}"
              f"{r['close']:>9.2f}"
              f"{body:>+8.2f}"
              f"{r['total_vol']:>10,.0f}"
              f"{r['delta_old']:>+11,.0f}"
              f"{r['delta_new']:>+11,.0f}"
              f"{flag:>8}")

    if len(disagree) > 0:
        print(f"\n  *FLIP* = sign disagreement between methods")
        print(f"          (i.e. close-vs-open says positive but OHLC says negative,")
        print(f"           or vice versa). These bars are where the new method")
        print(f"           would produce a different delta-mismatch decision.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*70}")
    print(f"  DELTA METHOD DIAGNOSTIC")
    print(f"{'═'*70}")

    df_1h = load_both_deltas()
    df_1h = compare_methods(df_1h)
    show_target_week(df_1h, TARGET_WEEK_START, TARGET_WEEK_END)

    # Save full comparison to CSV
    out_dir = r"C:\Trading\InitialBalanceAuctionFade\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "delta_comparison.csv")
    df_1h.to_csv(out_path, index=False)

    print(f"\n{'═'*70}")
    print(f"  Full comparison saved to:")
    print(f"  {out_path}")
    print(f"{'═'*70}\n")

    print("Next steps:")
    print("  1. Review the TOP 20 DIVERGENT BARS table — do the directions")
    print("     of disagreement match your intuition? E.g., long-upper-wick")
    print("     green bars should flip toward negative delta under the new")
    print("     method.")
    print("  2. Check the TARGET WEEK section — does the missing-trade bar")
    print("     show a sign flip? If yes, the new method would catch it.")
    print("  3. If sign-flip rate is high (>15%) AND the directions look")
    print("     intuitive, switch Config.DELTA_METHOD = 'ohlc_decomposition'")
    print("     and re-run the strategy. Compare results.")
    print("  4. If sign-flip rate is low (<5%) OR directions look random,")
    print("     the methods are mostly equivalent and the missing-trade")
    print("     issue is somewhere else (state machine logic, not delta).\n")


if __name__ == "__main__":
    main()
