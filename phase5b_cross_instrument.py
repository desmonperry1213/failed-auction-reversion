# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5b — CROSS-INSTRUMENT REPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per IB_Auction_Fade_Framework.pdf Section 4.4 and Section 5.
#
# Runs the IDENTICAL pre-registered state machine on NQ, GC, ZN with no
# per-instrument parameter tuning. Loads ES results from the existing
# trades.csv. Combines all four p-values via Fisher's method to produce
# the joint test statistic.
#
# IDENTICAL PARAMETERS (locked 2026-05-05):
#     SL_ATR_MULT_TPO    = 1.578
#     MIN_PROBE_DIST_ATR = 0.40
#     VELOCITY_THRESHOLD = 2.21
#     VA_THRESHOLD       = 0.70
#
# PER-INSTRUMENT OVERRIDES (mechanical only — not pre-registered parameters):
#     TICK_SIZE   varies by instrument (price-unit math)
#     POINT_VALUE varies by instrument (P&L scaling only — does NOT affect
#                 trade decisions or R-multiples; included for $ reporting)
#
# Outputs (saved to results\cross_instrument\):
#     trades_NQ.csv  trades_GC.csv  trades_ZN.csv
#     cross_instrument_summary.txt
#     cross_instrument_bootstrap.png   (4-panel: ES, NQ, GC, ZN)
#     fisher_combined.txt
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from scipy.stats import chi2

from strategy import (
    Config,
    compute_delta_ohlc_decomposition,
)
from tpo_state_machine import TPOStateMachine, trades_to_dataframe
from phase5_bootstrap import bootstrap_avg_r, N_BOOTSTRAP, SEED


# ── Paths ─────────────────────────────────────────────────────────────────────
CROSS_CSV_PATH = (
    r"C:\Trading\InitialBalanceAuctionFade\data"
    r"\OHLCV-1M CME Globex Cross Instrument"
    r"\OHLCV-1M CME Globex Cross Instrument Data"
    r"\OHLCV-1M CME Globex Cross Intrument.csv"
)
ES_TRADES_CSV = r"C:\Trading\InitialBalanceAuctionFade\results\confirmation_backtest\trades.csv"
OUT_DIR       = r"C:\Trading\InitialBalanceAuctionFade\results\cross_instrument"


# ── Per-instrument configuration ──────────────────────────────────────────────
# These are the ONLY values that vary per instrument. Strategy parameters
# are locked and unchanged.
INSTRUMENT_CONFIG = {
    'NQ': {
        'tick_size'   : 0.25,
        'point_value' : 20.0,    # E-mini Nasdaq, $20/point
        'description' : 'E-mini Nasdaq 100',
    },
    'GC': {
        'tick_size'   : 0.10,
        'point_value' : 100.0,   # Gold, $100/oz × 1 contract = $100/$1 move
        'description' : 'Gold Futures',
    },
    'ZN': {
        'tick_size'   : 1/64,    # = 0.015625
        'point_value' : 1000.0,  # 10Y Note, $1000 per 1.0 of decimal price
        'description' : '10-Year Treasury Note',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE PER INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_front_month(df):
    """Same logic as strategy.py — pick the highest-daily-volume symbol per day."""
    df = df.copy()
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
    return df


def aggregate_1h_from_1m(df_1m):
    """Aggregate 1M bars to 1H bars. No information loss."""
    df = df_1m.copy()
    df['hour_bucket'] = df['ts_event'].dt.floor('h')
    df_1h = df.groupby('hour_bucket').agg(
        open   = ('open',   'first'),
        high   = ('high',   'max'),
        low    = ('low',    'min'),
        close  = ('close',  'last'),
        volume = ('volume', 'sum'),
    ).reset_index().rename(columns={'hour_bucket': 'datetime'})
    return df_1h


def prepare_instrument(df_raw_full, prefix):
    """
    Filter the combined CSV to one instrument, drop spreads, build front-month,
    aggregate 1H, compute OHLC-decomposition delta, return df_1h and df_1m.
    """
    print(f"\n  Processing {prefix}...")
    CT = pytz.timezone(Config.TIMEZONE)

    # Filter by symbol prefix
    mask = df_raw_full['symbol'].str.startswith(prefix)
    df = df_raw_full.loc[mask].copy()
    print(f"    Raw {prefix} rows         : {len(df):,}")

    # Drop spread contracts (contain '-')
    df = df[~df['symbol'].str.contains('-', na=False)]
    print(f"    After dropping spreads   : {len(df):,}")

    # Timezone convert (data is UTC)
    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')
    df['ts_event'] = df['ts_event'].dt.tz_convert(CT)

    # Front-month continuous
    df_1m = build_front_month(df)
    df_1m = df_1m.sort_values('ts_event').reset_index(drop=True)
    print(f"    Front-month 1M rows      : {len(df_1m):,}")

    # OHLC-decomposition delta on 1M
    up, down = compute_delta_ohlc_decomposition(df_1m)
    df_1m['delta_1m'] = up - down

    # Aggregate 1H
    df_1h = aggregate_1h_from_1m(df_1m.rename(columns={'ts_event': 'ts_event'}))
    print(f"    Aggregated 1H rows       : {len(df_1h):,}")

    # Hourly delta = sum of constituent 1M deltas
    df_1m_for_delta = df_1m[['ts_event', 'delta_1m']].copy()
    df_1m_for_delta['hour_bucket'] = df_1m_for_delta['ts_event'].dt.floor('h')
    delta_1h = df_1m_for_delta.groupby('hour_bucket')['delta_1m'].sum().reset_index()
    delta_1h.columns = ['datetime', 'delta']
    df_1h = df_1h.merge(delta_1h, on='datetime', how='left')
    df_1h['delta'] = df_1h['delta'].fillna(0.0)

    # Standardize df_1m columns to match what TPOStateMachine expects
    df_1m = df_1m.rename(columns={'ts_event': 'datetime'})
    df_1m = df_1m[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    return df_1h, df_1m


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_instrument(df_raw_full, prefix, inst_cfg):
    """Process the data and run the state machine for a single instrument."""
    print(f"\n  {'─'*68}")
    print(f"  {prefix} — {inst_cfg['description']}")
    print(f"    tick_size = {inst_cfg['tick_size']}, "
          f"point_value = ${inst_cfg['point_value']:.2f}")
    print(f"  {'─'*68}")

    df_1h, df_1m = prepare_instrument(df_raw_full, prefix)

    # Override per-instrument mechanical params; restore on exit
    saved_tick   = Config.TICK_SIZE
    saved_pv     = Config.POINT_VALUE
    try:
        Config.TICK_SIZE   = inst_cfg['tick_size']
        Config.POINT_VALUE = inst_cfg['point_value']

        print(f"    Running TPO state machine...")
        t0 = time.time()
        sm = TPOStateMachine(df_1h, df_1m, contracts=1, verbose=False)
        trades = sm.run()
        print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(trades):,}")
    finally:
        Config.TICK_SIZE   = saved_tick
        Config.POINT_VALUE = saved_pv

    # Save per-instrument trades
    df_trades = trades_to_dataframe(trades)
    out_csv = os.path.join(OUT_DIR, f"trades_{prefix}.csv")
    df_trades.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")
    return df_trades


# ═══════════════════════════════════════════════════════════════════════════════
# FISHER'S COMBINED TEST
# ═══════════════════════════════════════════════════════════════════════════════

def fishers_combined(p_values):
    """
    Fisher's method for combining independent p-values.
        chi-squared = -2 × Σ ln(p_i)
        df = 2k where k = number of tests
    Returns chi-squared statistic, df, and combined p-value.
    """
    p = np.asarray(p_values, dtype=float)
    p = p[(p > 0) & (p <= 1)]  # guard against degenerate values
    if len(p) < 2:
        return float('nan'), 0, float('nan')
    chi_sq = -2.0 * np.sum(np.log(p))
    df     = 2 * len(p)
    p_comb = float(chi2.sf(chi_sq, df))   # survival function = 1 - CDF
    return float(chi_sq), df, p_comb


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_row(label, n, obs, ci_low, ci_high, p):
    sig = "*" if p < 0.05 else " "
    return (f"  {label:<20}{n:>5}  {obs:>+8.4f}  "
            f"[{ci_low:>+7.4f}, {ci_high:>+7.4f}]  {p:>7.4f}{sig}")


def plot_cross_instrument(per_instrument, out_path):
    """4-panel figure: bootstrap distribution of avg-R for each instrument."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor='#131722')
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor('#1e222d')
        for sp in ax.spines.values():
            sp.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

    instruments = ['ES', 'NQ', 'GC', 'ZN']
    for i, inst in enumerate(instruments):
        ax = axes[i]
        d  = per_instrument[inst]
        if d['n'] == 0:
            ax.text(0.5, 0.5, f"{inst}\n(no trades)", color='#d1d4dc',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.hist(d['resamples'], bins=60, color='#5dade2',
                edgecolor='#1e222d', alpha=0.85)
        ax.axvline(d['obs'],     color='#f1c40f', linestyle='-',  linewidth=1.8,
                   label=f"obs = {d['obs']:+.4f}")
        ax.axvline(d['ci_low'],  color='#bb6bd9', linestyle='--', linewidth=1.0,
                   label=f"CI low  = {d['ci_low']:+.4f}")
        ax.axvline(d['ci_high'], color='#bb6bd9', linestyle='--', linewidth=1.0,
                   label=f"CI high = {d['ci_high']:+.4f}")
        ax.axvline(0, color='#d1d4dc', linewidth=0.6, alpha=0.4)
        sig = "*" if d['p_value'] < 0.05 else ""
        ax.set_title(f"{inst}  (n={d['n']})  p={d['p_value']:.4f}{sig}", fontsize=11)
        ax.set_xlabel('avg-R'); ax.set_ylabel('count')
        ax.legend(facecolor='#1e222d', edgecolor='#2a2e39',
                  labelcolor='#d1d4dc', fontsize=7)

    fig.suptitle('Phase 5b — Cross-Instrument Bootstrap (identical pre-registered parameters)',
                 color='#d1d4dc', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*72}")
    print("  PHASE 5b — CROSS-INSTRUMENT REPLICATION")
    print(f"{'='*72}")
    print(f"\n  PRE-REGISTERED PARAMETERS (unchanged from ES):")
    print(f"    SL_ATR_MULT_TPO     = {Config.SL_ATR_MULT_TPO}")
    print(f"    MIN_PROBE_DIST_ATR  = {Config.MIN_PROBE_DIST_ATR}")
    print(f"    VELOCITY_THRESHOLD  = {Config.VELOCITY_THRESHOLD}")
    print(f"    VA_THRESHOLD        = {Config.VA_THRESHOLD}")

    # Force OHLC-decomposition delta
    saved_delta = getattr(Config, 'DELTA_METHOD', 'close_vs_open')
    Config.DELTA_METHOD = 'ohlc_decomposition'

    # Load combined CSV (this is the slow step)
    print(f"\n  Loading combined CSV (~22M rows)...")
    t0 = time.time()
    df_raw = pd.read_csv(CROSS_CSV_PATH, parse_dates=['ts_event'])
    print(f"    Loaded {len(df_raw):,} rows in {time.time()-t0:.0f}s")
    print(f"    Columns: {list(df_raw.columns)}")

    # Run each instrument
    per_instrument_trades = {}
    try:
        for prefix in ['NQ', 'GC', 'ZN']:
            df_trades = run_instrument(df_raw, prefix, INSTRUMENT_CONFIG[prefix])
            per_instrument_trades[prefix] = df_trades
    finally:
        Config.DELTA_METHOD = saved_delta

    # Free the giant raw frame before bootstrap
    del df_raw

    # Load existing ES trades
    print(f"\n  Loading existing ES trades from {ES_TRADES_CSV}...")
    if not os.path.exists(ES_TRADES_CSV):
        print(f"    ERROR: file not found. Run run_tpo_confirmation.py first.")
        return
    df_es = pd.read_csv(ES_TRADES_CSV)
    per_instrument_trades['ES'] = df_es
    print(f"    Loaded {len(df_es):,} ES trades")

    # ── Bootstrap each instrument ────────────────────────────────────────────
    print(f"\n  Running bootstrap for each instrument ({N_BOOTSTRAP:,} resamples each)...")
    per_instrument_boot = {}
    for inst in ['ES', 'NQ', 'GC', 'ZN']:
        df_t = per_instrument_trades[inst]
        if len(df_t) == 0:
            per_instrument_boot[inst] = {
                'n': 0, 'obs': 0.0, 'resamples': np.array([]),
                'ci_low': 0.0, 'ci_high': 0.0, 'p_value': float('nan'),
            }
            continue
        per_instrument_boot[inst] = bootstrap_avg_r(df_t['r_multiple'])

    # ── Fisher's combined test ───────────────────────────────────────────────
    p_values_for_fisher = [per_instrument_boot[inst]['p_value']
                           for inst in ['ES', 'NQ', 'GC', 'ZN']
                           if per_instrument_boot[inst]['n'] > 0
                           and np.isfinite(per_instrument_boot[inst]['p_value'])
                           and per_instrument_boot[inst]['p_value'] > 0]
    chi_sq, df_fisher, p_combined = fishers_combined(p_values_for_fisher)

    # ── Build summary report ─────────────────────────────────────────────────
    lines = []
    lines.append("="*72)
    lines.append("PHASE 5b — CROSS-INSTRUMENT REPLICATION RESULTS")
    lines.append("="*72)
    lines.append(f"  Pre-registered params : SL_ATR_MULT_TPO={Config.SL_ATR_MULT_TPO}  "
                 f"MIN_PROBE_DIST_ATR={Config.MIN_PROBE_DIST_ATR}  "
                 f"VELOCITY_THRESHOLD={Config.VELOCITY_THRESHOLD}")
    lines.append(f"  Delta method          : ohlc_decomposition (forced)")
    lines.append(f"  Bootstrap resamples   : {N_BOOTSTRAP:,} per test, seed {SEED}")
    lines.append("")
    lines.append("  TABLE 3 — Per-Instrument Bootstrap Results")
    lines.append("")
    lines.append(f"  {'Instrument':<20}{'n':>5}  {'avg-R':>8}  "
                 f"{'95% CI':<22}  {'p-value':>8}")
    lines.append("  " + "-"*70)
    for inst in ['ES', 'NQ', 'GC', 'ZN']:
        d = per_instrument_boot[inst]
        if d['n'] == 0:
            lines.append(f"  {inst:<20}(no trades)")
            continue
        lines.append(fmt_row(
            f"{inst} ({INSTRUMENT_CONFIG.get(inst, {}).get('description', 'ES — S&P 500')})",
            d['n'], d['obs'], d['ci_low'], d['ci_high'], d['p_value']
        ))
    lines.append("")
    lines.append("="*72)
    lines.append("FISHER'S COMBINED TEST  (framework Section 4.4)")
    lines.append("="*72)
    lines.append(f"  Method     : chi-squared = -2 × Σ ln(p_i)")
    lines.append(f"  k tests    : {len(p_values_for_fisher)}")
    lines.append(f"  df         : {df_fisher}")
    lines.append(f"  Chi-squared: {chi_sq:.4f}")
    lines.append(f"  Combined p : {p_combined:.6f}")
    lines.append(f"  Significant: {'YES (reject joint null)' if p_combined < 0.05 else 'no (cannot reject joint null)'}")
    lines.append("")
    lines.append(f"  Joint null hypothesis: all four instruments have zero edge simultaneously.")
    lines.append(f"  A combined p < 0.05 is harder to achieve by chance than any single test")
    lines.append(f"  and constitutes strong evidence the mechanism is real across asset classes.")
    lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)

    # Save reports
    out_summary = os.path.join(OUT_DIR, "cross_instrument_summary.txt")
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n  Saved: {out_summary}")

    out_fisher = os.path.join(OUT_DIR, "fisher_combined.txt")
    with open(out_fisher, 'w', encoding='utf-8') as f:
        f.write(
            f"Fisher's Combined Test\n"
            f"  Inputs (per-instrument p-values): {dict(zip(['ES','NQ','GC','ZN'], [per_instrument_boot[i]['p_value'] for i in ['ES','NQ','GC','ZN']]))}\n"
            f"  Chi-squared : {chi_sq}\n"
            f"  df          : {df_fisher}\n"
            f"  Combined p  : {p_combined}\n"
            f"  Reject joint null at 0.05 ? : {'YES' if p_combined < 0.05 else 'no'}\n"
        )
    print(f"  Saved: {out_fisher}")

    # ── Cross-instrument figure ──────────────────────────────────────────────
    fig_path = os.path.join(OUT_DIR, "cross_instrument_bootstrap.png")
    plot_cross_instrument(per_instrument_boot, fig_path)
    print(f"  Saved: {fig_path}")

    print(f"\n{'='*72}")
    print("  Phase 5b complete.")
    print(f"{'='*72}")
    print()
    print("  Decision tree:")
    print("    - Fisher combined p < 0.05  → REJECT joint null. Mechanism appears real")
    print("        across asset classes. Strong basis for Paper 1.")
    print("    - Fisher p ≥ 0.05 but some instruments ind. sig → CONDITIONAL evidence")
    print("        — paper still publishable but more nuanced framing.")
    print("    - Fisher p ≥ 0.05 AND no individual instrument sig → null cannot be")
    print("        rejected. Honest negative result. Paper reframes as a study of")
    print("        market efficiency dynamics rather than profitable strategy.")
    print()


if __name__ == "__main__":
    main()
