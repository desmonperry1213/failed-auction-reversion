# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5b (5M) — CROSS-INSTRUMENT REPLICATION AT 5M RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per IB_Auction_Fade_Framework.pdf Section 4.4 and Section 5, with the
# methodologically-motivated 5M re-pre-registration.
#
# Runs the IDENTICAL 5M state machine on NQ, GC, ZN with no per-instrument
# tuning. Loads ES 5M results from the existing trades.csv. Combines all four
# p-values via Fisher's method.
#
# 5M RE-PRE-REGISTERED PARAMETERS (locked 2026-05-06):
#     SL_ATR_MULT_5M         = 4.544
#     MIN_PROBE_DIST_ATR_5M  = 1.045
#     VELOCITY_THRESHOLD_5M  = 3.808
#     VA_THRESHOLD           = 0.70  (unchanged from 1H)
#
# PER-INSTRUMENT OVERRIDES (mechanical only — not pre-registered):
#     TICK_SIZE   varies by instrument
#     POINT_VALUE varies by instrument (P&L scaling only)
#
# Outputs (saved to results\cross_instrument_5m\):
#     trades_NQ.csv  trades_GC.csv  trades_ZN.csv
#     cross_instrument_summary_5m.txt
#     cross_instrument_bootstrap_5m.png   (4-panel: ES, NQ, GC, ZN)
#     fisher_combined_5m.txt
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from scipy.stats import chi2

from strategy import Config, compute_delta_ohlc_decomposition
from tpo_state_machine import TPOStateMachine, trades_to_dataframe
from phase5_bootstrap import bootstrap_avg_r, N_BOOTSTRAP, SEED


# ── Paths ─────────────────────────────────────────────────────────────────────
CROSS_CSV_PATH = (
    r"C:\Trading\InitialBalanceAuctionFade\data"
    r"\OHLCV-1M CME Globex Cross Instrument"
    r"\OHLCV-1M CME Globex Cross Instrument Data"
    r"\OHLCV-1M CME Globex Cross Intrument.csv"
)
ES_5M_TRADES_CSV = r"C:\Trading\InitialBalanceAuctionFade\results\confirmation_backtest_5m\trades.csv"
OUT_DIR          = r"C:\Trading\InitialBalanceAuctionFade\results\cross_instrument_5m"

TIMEFRAME_FREQ   = '5min'
BARS_PER_DAY     = 276


# ── Per-instrument configuration ──────────────────────────────────────────────
INSTRUMENT_CONFIG = {
    'NQ': {
        'tick_size'   : 0.25,
        'point_value' : 20.0,
        'description' : 'E-mini Nasdaq 100',
    },
    'GC': {
        'tick_size'   : 0.10,
        'point_value' : 100.0,
        'description' : 'Gold Futures',
    },
    'ZN': {
        'tick_size'   : 1/64,
        'point_value' : 1000.0,
        'description' : '10-Year Treasury Note',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE PER INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_front_month(df):
    """Pick the highest-daily-volume symbol per day."""
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


def aggregate_to_5m(df_1m):
    """Aggregate 1M bars to 5M with summed delta."""
    df = df_1m.copy()
    df['bucket'] = df['datetime'].dt.floor(TIMEFRAME_FREQ)
    bars = df.groupby('bucket').agg(
        open   = ('open',     'first'),
        high   = ('high',     'max'),
        low    = ('low',      'min'),
        close  = ('close',    'last'),
        volume = ('volume',   'sum'),
        delta  = ('delta_1m', 'sum'),
    ).reset_index().rename(columns={'bucket': 'datetime'})
    return bars


def prepare_instrument(df_raw_full, prefix):
    """
    Filter to one instrument, drop spreads, build front-month, compute
    OHLC-decomposition delta on 1M, aggregate to 5M.
    Returns (df_5m, df_1m_indexed_for_engine).
    """
    print(f"\n  Processing {prefix}...")
    CT = pytz.timezone(Config.TIMEZONE)

    mask = df_raw_full['symbol'].str.startswith(prefix)
    df = df_raw_full.loc[mask].copy()
    print(f"    Raw {prefix} rows         : {len(df):,}")

    df = df[~df['symbol'].str.contains('-', na=False)]
    print(f"    After dropping spreads   : {len(df):,}")

    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')
    df['ts_event'] = df['ts_event'].dt.tz_convert(CT)

    df_1m = build_front_month(df)
    df_1m = df_1m.sort_values('ts_event').reset_index(drop=True)
    print(f"    Front-month 1M rows      : {len(df_1m):,}")

    print(f"    Computing OHLC-decomposition delta on 1M...")
    up, down = compute_delta_ohlc_decomposition(df_1m)
    df_1m['delta_1m'] = up - down

    df_1m = df_1m.rename(columns={'ts_event': 'datetime'})
    df_1m = df_1m[['datetime', 'open', 'high', 'low', 'close', 'volume', 'delta_1m']]

    df_5m = aggregate_to_5m(df_1m)
    print(f"    Aggregated 5M rows       : {len(df_5m):,}")

    return df_5m, df_1m


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_instrument(df_raw_full, prefix, inst_cfg):
    print(f"\n  {'─'*68}")
    print(f"  {prefix} — {inst_cfg['description']}")
    print(f"    tick_size = {inst_cfg['tick_size']}, "
          f"point_value = ${inst_cfg['point_value']:.2f}")
    print(f"  {'─'*68}")

    df_5m, df_1m = prepare_instrument(df_raw_full, prefix)

    saved_tick = Config.TICK_SIZE
    saved_pv   = Config.POINT_VALUE
    try:
        Config.TICK_SIZE   = inst_cfg['tick_size']
        Config.POINT_VALUE = inst_cfg['point_value']

        print(f"    Running TPO state machine at 5M with 5M params...")
        t0 = time.time()
        sm = TPOStateMachine(
            df_1h        = df_5m,
            df_1m        = df_1m,
            contracts    = 1,
            verbose      = False,
            gate_enabled = True,
            entry_mode   = 'retest',
            bar_freq     = TIMEFRAME_FREQ,
            bars_per_day = BARS_PER_DAY,
            param_set    = '5m',
        )
        trades = sm.run()
        print(f"    Done in {time.time()-t0:.0f}s.  Trades: {len(trades):,}")
    finally:
        Config.TICK_SIZE   = saved_tick
        Config.POINT_VALUE = saved_pv

    df_trades = trades_to_dataframe(trades)
    out_csv = os.path.join(OUT_DIR, f"trades_{prefix}.csv")
    df_trades.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")
    return df_trades


# ═══════════════════════════════════════════════════════════════════════════════
# FISHER'S COMBINED TEST
# ═══════════════════════════════════════════════════════════════════════════════

def fishers_combined(p_values):
    p = np.asarray(p_values, dtype=float)
    p = p[(p > 0) & (p <= 1)]
    if len(p) < 2:
        return float('nan'), 0, float('nan')
    chi_sq = -2.0 * np.sum(np.log(p))
    df     = 2 * len(p)
    p_comb = float(chi2.sf(chi_sq, df))
    return float(chi_sq), df, p_comb


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_row(label, n, obs, ci_low, ci_high, p):
    sig = "*" if p < 0.05 else " "
    return (f"  {label:<28}{n:>6}  {obs:>+8.4f}  "
            f"[{ci_low:>+7.4f}, {ci_high:>+7.4f}]  {p:>7.4f}{sig}")


def plot_cross_instrument(per_instrument, out_path):
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
        ax.set_title(f"{inst}  (n={d['n']:,})  p={d['p_value']:.4f}{sig}", fontsize=11)
        ax.set_xlabel('avg-R'); ax.set_ylabel('count')
        ax.legend(facecolor='#1e222d', edgecolor='#2a2e39',
                  labelcolor='#d1d4dc', fontsize=7)

    fig.suptitle('Phase 5b (5M) — Cross-Instrument Bootstrap '
                 '(5M re-pre-registered parameters)',
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
    print("  PHASE 5b (5M) — CROSS-INSTRUMENT REPLICATION")
    print(f"{'='*72}")
    print(f"\n  5M RE-PRE-REGISTERED PARAMETERS (unchanged from ES):")
    print(f"    SL_ATR_MULT_5M         = {Config.SL_ATR_MULT_5M}")
    print(f"    MIN_PROBE_DIST_ATR_5M  = {Config.MIN_PROBE_DIST_ATR_5M}")
    print(f"    VELOCITY_THRESHOLD_5M  = {Config.VELOCITY_THRESHOLD_5M}")
    print(f"    VA_THRESHOLD           = {Config.VA_THRESHOLD}")

    saved_delta = getattr(Config, 'DELTA_METHOD', 'close_vs_open')
    Config.DELTA_METHOD = 'ohlc_decomposition'

    print(f"\n  Loading combined CSV (~22M rows)...")
    t0 = time.time()
    df_raw = pd.read_csv(CROSS_CSV_PATH, parse_dates=['ts_event'])
    print(f"    Loaded {len(df_raw):,} rows in {time.time()-t0:.0f}s")

    per_instrument_trades = {}
    try:
        for prefix in ['NQ', 'GC', 'ZN']:
            df_trades = run_instrument(df_raw, prefix, INSTRUMENT_CONFIG[prefix])
            per_instrument_trades[prefix] = df_trades
    finally:
        Config.DELTA_METHOD = saved_delta

    del df_raw

    print(f"\n  Loading existing ES 5M trades from {ES_5M_TRADES_CSV}...")
    if not os.path.exists(ES_5M_TRADES_CSV):
        print(f"    ERROR: file not found. Run run_tpo_confirmation_5m.py first.")
        return
    df_es = pd.read_csv(ES_5M_TRADES_CSV)
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
    lines.append("PHASE 5b (5M) — CROSS-INSTRUMENT REPLICATION RESULTS")
    lines.append("="*72)
    lines.append(f"  5M re-pre-registered  : SL_ATR_MULT_5M={Config.SL_ATR_MULT_5M}  "
                 f"MIN_PROBE_DIST_ATR_5M={Config.MIN_PROBE_DIST_ATR_5M}  "
                 f"VELOCITY_THRESHOLD_5M={Config.VELOCITY_THRESHOLD_5M}")
    lines.append(f"  Delta method          : ohlc_decomposition (forced)")
    lines.append(f"  Bootstrap resamples   : {N_BOOTSTRAP:,} per test, seed {SEED}")
    lines.append("")
    lines.append("  TABLE 4 — Per-Instrument Bootstrap Results (5M)")
    lines.append("")
    lines.append(f"  {'Instrument':<28}{'n':>6}  {'avg-R':>8}  "
                 f"{'95% CI':<22}  {'p-value':>8}")
    lines.append("  " + "-"*72)
    for inst in ['ES', 'NQ', 'GC', 'ZN']:
        d = per_instrument_boot[inst]
        if d['n'] == 0:
            lines.append(f"  {inst:<28}(no trades)")
            continue
        desc = (INSTRUMENT_CONFIG.get(inst, {}).get('description', 'ES — S&P 500')
                if inst != 'ES' else 'ES — S&P 500')
        lines.append(fmt_row(
            f"{inst} ({desc})",
            d['n'], d['obs'], d['ci_low'], d['ci_high'], d['p_value']
        ))
    lines.append("")
    lines.append("="*72)
    lines.append("FISHER'S COMBINED TEST  (framework Section 4.4) — 5M")
    lines.append("="*72)
    lines.append(f"  Method     : chi-squared = -2 × Σ ln(p_i)")
    lines.append(f"  k tests    : {len(p_values_for_fisher)}")
    lines.append(f"  df         : {df_fisher}")
    lines.append(f"  Chi-squared: {chi_sq:.4f}")
    lines.append(f"  Combined p : {p_combined:.6f}")
    lines.append(f"  Significant: "
                 f"{'YES (reject joint null)' if p_combined < 0.05 else 'no (cannot reject joint null)'}")
    lines.append("")
    lines.append("  Joint null: all four instruments have zero edge simultaneously.")
    lines.append("  Combined p < 0.05 is harder to achieve by chance than any single test")
    lines.append("  and constitutes strong evidence the mechanism is real across asset classes.")
    lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)

    out_summary = os.path.join(OUT_DIR, "cross_instrument_summary_5m.txt")
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n  Saved: {out_summary}")

    out_fisher = os.path.join(OUT_DIR, "fisher_combined_5m.txt")
    with open(out_fisher, 'w', encoding='utf-8') as f:
        p_dict = {i: per_instrument_boot[i]['p_value'] for i in ['ES','NQ','GC','ZN']}
        f.write(
            f"Fisher's Combined Test (5M)\n"
            f"  Inputs: {p_dict}\n"
            f"  Chi-squared : {chi_sq}\n"
            f"  df          : {df_fisher}\n"
            f"  Combined p  : {p_combined}\n"
            f"  Reject joint null at 0.05 ? : {'YES' if p_combined < 0.05 else 'no'}\n"
        )
    print(f"  Saved: {out_fisher}")

    fig_path = os.path.join(OUT_DIR, "cross_instrument_bootstrap_5m.png")
    plot_cross_instrument(per_instrument_boot, fig_path)
    print(f"  Saved: {fig_path}")

    print(f"\n{'='*72}")
    print("  Phase 5b (5M) complete.")
    print(f"{'='*72}")
    print()
    print("  Final decision tree:")
    print("    - Fisher combined p < 0.05  → REJECT joint null. Cross-asset")
    print("        replication confirms the mechanism. STRONG paper.")
    print("    - Fisher p ≥ 0.05 but ES individually sig → ES-specific finding.")
    print("        Still publishable, more nuanced framing required.")
    print("    - Fisher p ≥ 0.05 AND ES not individually sig → null cannot be")
    print("        rejected on either test. Honest negative result.")
    print()


if __name__ == "__main__":
    main()
