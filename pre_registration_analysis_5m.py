# ═══════════════════════════════════════════════════════════════════════════════
# PRE-REGISTRATION ANALYSIS — 5M TIMEFRAME (re-pre-registration)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Re-derives the three locked parameters (SL_ATR_MULT, MIN_PROBE_DIST_ATR,
# VELOCITY_THRESHOLD) from 5M probe distributions on the full 2015–2026 ES
# dataset. Replaces the 1H pre-registration with a methodologically motivated
# 5M version, after the multi-timeframe diagnostic established that 5M
# resolves the bar-collapse artifact present at 1H.
#
# IMPORTANT METHODOLOGICAL NOTE:
# This is a re-pre-registration. The original 1H values remain on permanent
# record. The 5M values become a SECOND pre-registered set. Both sets will
# be reported transparently in the paper. The decision to switch was based
# on the diagnostic chain — original null result → filter funnel → no-retest
# → multi-timeframe — not on the 5M strategy result (which has not been
# computed under new parameters yet).
#
# Outputs (saved to results\pre_registration_5m\):
#     probe_distribution_5m.csv
#     probe_distributions_5m.png
#     stability_check_5m.png
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

from strategy import Config, compute_delta_ohlc_decomposition
from tpo_engine import TPOEngine


# ── Configuration ─────────────────────────────────────────────────────────────
ANALYSIS_START = "2015-01-01"
ANALYSIS_END   = "2026-12-31"
TIMEFRAME_FREQ = '5min'
BARS_PER_DAY   = 276       # 23 hours × 12 (5M bars per hour)
ATR_PERIOD     = 14        # framework default — keep at 14 bars regardless of TF
ROLLING_VOL_DAYS = 60      # framework default

OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\pre_registration_5m"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_es_1m_data():
    """Load ES 1M (front-month, US/Central, with OHLC-decomposition delta)."""
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
    print(f"  Computing OHLC-decomposition delta on 1M...")
    up, down = compute_delta_ohlc_decomposition(df)
    df['delta_1m'] = up - down

    # Range filter
    start = pd.Timestamp(ANALYSIS_START, tz=CT)
    end   = pd.Timestamp(ANALYSIS_END,   tz=CT)
    df    = df[(df['ts_event'] >= start) & (df['ts_event'] < end)].reset_index(drop=True)

    df = df.rename(columns={'ts_event': 'datetime'})
    print(f"  In range    : {len(df):,}")
    return df


def aggregate_to_5m(df_1m):
    """Aggregate 1M bars to 5M with summed delta."""
    print(f"\nAggregating 1M → 5M bars...")
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
    print(f"  5M bars     : {len(bars):,}")
    return bars


def compute_atr(df, period=14):
    """Standard 14-period ATR on bars."""
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


def compute_rolling_vol_median(df, days, bars_per_day):
    """Rolling N-day median of bar volume."""
    window_bars = max(int(days * bars_per_day), 100)
    return df['volume'].rolling(window_bars, min_periods=100).median().values


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  PRE-REGISTRATION ANALYSIS — 5M TIMEFRAME")
    print(f"  {ANALYSIS_START} → {ANALYSIS_END}")
    print(f"{'='*72}\n")

    # ── Load and aggregate ───────────────────────────────────────────────────
    df_1m  = load_es_1m_data()
    df_5m  = aggregate_to_5m(df_1m)

    df_5m['atr']         = compute_atr(df_5m, period=ATR_PERIOD)
    df_5m['rolling_vol'] = compute_rolling_vol_median(df_5m, days=ROLLING_VOL_DAYS,
                                                      bars_per_day=BARS_PER_DAY)
    df_5m['range']       = df_5m['high'] - df_5m['low']

    median_volume = float(df_5m['volume'].median())
    print(f"\n  Median 5M volume (global) : {median_volume:,.0f}")
    print(f"  ATR period                : {ATR_PERIOD} bars")
    print(f"  Rolling vol window        : {ROLLING_VOL_DAYS} days × "
          f"{BARS_PER_DAY} bars/day = {ROLLING_VOL_DAYS*BARS_PER_DAY:,} bars\n")

    # ── Build 1M index for engine feeding ────────────────────────────────────
    print("  Building 1M index...")
    df_1m['bucket'] = df_1m['datetime'].dt.floor(TIMEFRAME_FREQ)
    m_index = {b: g.reset_index(drop=True)
               for b, g in df_1m.groupby('bucket')}
    print(f"    {len(m_index):,} 5M buckets indexed\n")

    # ── Walk engine, harvest probes ──────────────────────────────────────────
    print(f"  Running TPO engine over {len(df_5m):,} 5M bars...")
    engine = TPOEngine()
    probes = []
    t0 = time.time()
    n_bars = len(df_5m)

    for i in range(n_bars):
        row = df_5m.iloc[i]
        bar_minutes = m_index.get(row['datetime'])
        engine.add_bar(row, bar_minutes)

        if engine.weekly_vah is None or engine.weekly_val is None:
            continue
        atr_v = row['atr']
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue

        c = row['close']
        if c > engine.weekly_vah:
            direction     = 'above'
            probe_extreme = row['high']
            distance      = probe_extreme - engine.weekly_vah
        elif c < engine.weekly_val:
            direction     = 'below'
            probe_extreme = row['low']
            distance      = engine.weekly_val - probe_extreme
        else:
            continue

        if distance < 0:
            distance = 0.0
        distance_atr_mult = distance / atr_v

        bar_vol     = float(row['volume']) if np.isfinite(row['volume']) else 0.0
        rolling_vol = row['rolling_vol']
        if (np.isfinite(rolling_vol) and rolling_vol > 0 and bar_vol > 0):
            velocity_score = (row['range'] / atr_v) / (bar_vol / rolling_vol)
        else:
            velocity_score = float('nan')

        probes.append({
            'datetime'         : row['datetime'],
            'direction'        : direction,
            'close'            : float(c),
            'probe_extreme'    : float(probe_extreme),
            'vah'              : float(engine.weekly_vah),
            'val'              : float(engine.weekly_val),
            'poc'              : float(engine.weekly_poc),
            'atr'              : float(atr_v),
            'bar_range'        : float(row['range']),
            'bar_volume'       : bar_vol,
            'rolling_vol'      : float(rolling_vol) if np.isfinite(rolling_vol) else 0.0,
            'distance'         : float(distance),
            'distance_atr_mult': float(distance_atr_mult),
            'velocity_score'   : float(velocity_score),
        })

        if (i + 1) % 50_000 == 0:
            print(f"    [{i+1:>7,}/{n_bars:,}]  probes: {len(probes):>6,}  "
                  f"elapsed: {time.time()-t0:.0f}s")

    print(f"\n  Done in {time.time()-t0:.0f}s.  {len(probes):,} probe events.")

    df_probes = pd.DataFrame(probes)
    csv_path = os.path.join(OUT_DIR, "probe_distribution_5m.csv")
    df_probes.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    if len(df_probes) == 0:
        print("\n  [WARN] No probe events. Aborting analysis.")
        return

    # ── Summary statistics ───────────────────────────────────────────────────
    n_total = len(df_probes)
    n_above = int((df_probes['direction'] == 'above').sum())
    n_below = int((df_probes['direction'] == 'below').sum())

    print(f"\n{'─'*72}")
    print(f"  PROBE DISTRIBUTION SUMMARY (5M)")
    print(f"{'─'*72}")
    print(f"  Total probes      : {n_total:,}")
    print(f"  Above weekly VAH  : {n_above:,}  ({n_above/n_total*100:.1f}%)")
    print(f"  Below weekly VAL  : {n_below:,}  ({n_below/n_total*100:.1f}%)")

    dam = df_probes['distance_atr_mult'].values
    median_dam = float(np.median(dam))
    print(f"\n  Distance / ATR(14) percentiles:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = float(np.percentile(dam, pct))
        marker = "  ← median (= recommended SL_ATR_MULT_5M)" if pct == 50 else ""
        print(f"    {pct:>3d}th : {v:>7.3f}{marker}")

    vs_all    = df_probes['velocity_score'].values
    vs_finite = vs_all[np.isfinite(vs_all)]
    median_vs = float(np.median(vs_finite))
    print(f"\n  Velocity score percentiles  (n_finite={len(vs_finite):,}):")
    for pct in [1, 5, 10, 25, 50, 75, 85, 90, 95, 99]:
        v = float(np.percentile(vs_finite, pct))
        print(f"    {pct:>3d}th : {v:>7.3f}")

    # ── 4-panel diagnostic figure ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor='#131722')
    for ax in axes.flat:
        ax.set_facecolor('#1e222d')
        for sp in ax.spines.values():
            sp.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

    # Distance histogram
    dam_top = float(np.percentile(dam, 99))
    axes[0, 0].hist(dam[dam <= dam_top], bins=100, color='#5dade2',
                    edgecolor='#1e222d', alpha=0.85)
    axes[0, 0].axvline(median_dam, color='#f1c40f', linestyle='--', linewidth=1.8,
                       label=f'Median = {median_dam:.3f}')
    axes[0, 0].set_title(f'Probe Distance / ATR(14)  — 5M, n={len(dam):,}')
    axes[0, 0].set_xlabel('distance / ATR'); axes[0, 0].set_ylabel('count')
    axes[0, 0].legend(facecolor='#1e222d', edgecolor='#2a2e39',
                      labelcolor='#d1d4dc', fontsize=9)

    # Distance CDF
    s_dam = np.sort(dam)
    cdf   = np.arange(1, len(s_dam) + 1) / len(s_dam)
    axes[0, 1].plot(s_dam, cdf, color='#5dade2', linewidth=1.4)
    axes[0, 1].axvline(median_dam, color='#f1c40f', linestyle='--', linewidth=1.5,
                       label=f'50th pct = {median_dam:.3f}')
    for pct in [10, 25, 75, 90]:
        v = float(np.percentile(dam, pct))
        axes[0, 1].axvline(v, color='#5dade2', linestyle=':', linewidth=0.8, alpha=0.7)
        axes[0, 1].text(v, 0.05, f'{pct}th\n{v:.2f}', color='#d1d4dc',
                        fontsize=7, ha='center', alpha=0.8)
    axes[0, 1].set_title('Probe Distance / ATR(14) — CDF')
    axes[0, 1].set_xlabel('distance / ATR'); axes[0, 1].set_ylabel('cumulative fraction')
    axes[0, 1].set_xlim(0, dam_top)
    axes[0, 1].legend(facecolor='#1e222d', edgecolor='#2a2e39',
                      labelcolor='#d1d4dc', fontsize=9)

    # Velocity histogram
    vs_top = float(np.percentile(vs_finite, 99))
    axes[1, 0].hist(vs_finite[vs_finite <= vs_top], bins=100, color='#bb6bd9',
                    edgecolor='#1e222d', alpha=0.85)
    axes[1, 0].axvline(median_vs, color='#f1c40f', linestyle='--', linewidth=1.8,
                       label=f'Median = {median_vs:.3f}')
    axes[1, 0].set_title(f'Velocity Score  — 5M, n_finite={len(vs_finite):,}')
    axes[1, 0].set_xlabel('velocity_score'); axes[1, 0].set_ylabel('count')
    axes[1, 0].legend(facecolor='#1e222d', edgecolor='#2a2e39',
                      labelcolor='#d1d4dc', fontsize=9)

    # Velocity CDF
    s_vs  = np.sort(vs_finite)
    vs_cdf = np.arange(1, len(s_vs) + 1) / len(s_vs)
    axes[1, 1].plot(s_vs, vs_cdf, color='#bb6bd9', linewidth=1.4)
    axes[1, 1].axvline(median_vs, color='#f1c40f', linestyle='--', linewidth=1.5,
                       label=f'50th pct = {median_vs:.3f}')
    for pct in [75, 85, 90, 95]:
        v = float(np.percentile(vs_finite, pct))
        axes[1, 1].axvline(v, color='#bb6bd9', linestyle=':', linewidth=0.8, alpha=0.7)
        axes[1, 1].text(v, 0.05, f'{pct}th\n{v:.2f}', color='#d1d4dc',
                        fontsize=7, ha='center', alpha=0.8)
    axes[1, 1].set_title('Velocity Score — CDF')
    axes[1, 1].set_xlabel('velocity_score'); axes[1, 1].set_ylabel('cumulative fraction')
    axes[1, 1].set_xlim(0, vs_top)
    axes[1, 1].legend(facecolor='#1e222d', edgecolor='#2a2e39',
                      labelcolor='#d1d4dc', fontsize=9)

    fig.suptitle(f'5M Pre-Registration Distributions  —  {n_total:,} probes  —  '
                 f'{ANALYSIS_START} to {ANALYSIS_END}',
                 color='#d1d4dc', fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "probe_distributions_5m.png")
    fig.savefig(fig_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    print(f"\n  Saved: {fig_path}")
    plt.show()

    # ── Sub-period stability check ───────────────────────────────────────────
    df_probes['year'] = pd.to_datetime(df_probes['datetime'], utc=True).dt.year
    sub_periods = [
        ('2015-2017', (2015, 2017)),
        ('2018-2020', (2018, 2020)),
        ('2021-2023', (2021, 2023)),
        ('2024-2026', (2024, 2026)),
    ]
    sub_colors = ['#5dade2', '#26a69a', '#f1c40f', '#ef5350']

    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), facecolor='#131722')
    for ax in axes2:
        ax.set_facecolor('#1e222d')
        for sp in ax.spines.values():
            sp.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

    print(f"\n{'─'*72}")
    print(f"  SUB-PERIOD STABILITY CHECK (5M)")
    print(f"{'─'*72}")
    print(f"  {'Period':<12}{'n':>10}{'p50_dist':>12}{'p90_dist':>12}"
          f"{'p50_vel':>12}{'p90_vel':>12}")

    sub_summary = []
    for (label, (y0, y1)), color in zip(sub_periods, sub_colors):
        sub = df_probes[(df_probes['year'] >= y0) & (df_probes['year'] <= y1)]
        if len(sub) < 100:
            print(f"  {label:<12}{len(sub):>10}    (skipping — too few probes)")
            continue

        sub_dam = sub['distance_atr_mult'].values
        sub_vs  = sub['velocity_score'].values
        sub_vs  = sub_vs[np.isfinite(sub_vs)]

        s_d = np.sort(sub_dam); c_d = np.arange(1, len(s_d) + 1) / len(s_d)
        axes2[0].plot(s_d, c_d, color=color, linewidth=1.4, label=f'{label}  (n={len(sub):,})')
        s_v = np.sort(sub_vs);  c_v = np.arange(1, len(s_v) + 1) / len(s_v)
        axes2[1].plot(s_v, c_v, color=color, linewidth=1.4, label=f'{label}  (n={len(sub):,})')

        p50_d = float(np.percentile(sub_dam, 50))
        p90_d = float(np.percentile(sub_dam, 90))
        p50_v = float(np.percentile(sub_vs,  50))
        p90_v = float(np.percentile(sub_vs,  90))
        sub_summary.append((label, len(sub), p50_d, p90_d, p50_v, p90_v))
        print(f"  {label:<12}{len(sub):>10,}{p50_d:>12.3f}{p90_d:>12.3f}"
              f"{p50_v:>12.3f}{p90_v:>12.3f}")

    axes2[0].set_title('Distance / ATR — CDF by sub-period (5M)')
    axes2[0].set_xlabel('distance / ATR'); axes2[0].set_ylabel('cumulative fraction')
    axes2[0].set_xlim(0, dam_top)
    axes2[0].legend(facecolor='#1e222d', edgecolor='#2a2e39',
                    labelcolor='#d1d4dc', fontsize=8)

    axes2[1].set_title('Velocity Score — CDF by sub-period (5M)')
    axes2[1].set_xlabel('velocity_score'); axes2[1].set_ylabel('cumulative fraction')
    axes2[1].set_xlim(0, vs_top)
    axes2[1].legend(facecolor='#1e222d', edgecolor='#2a2e39',
                    labelcolor='#d1d4dc', fontsize=8)

    fig2.suptitle('5M Sub-Period Stability — overlap = stable, divergence = drift',
                  color='#d1d4dc', fontsize=12)
    fig2.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "stability_check_5m.png")
    fig2.savefig(fig2_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    print(f"\n  Saved: {fig2_path}")

    if len(sub_summary) >= 2:
        spread_d50 = max(s[2] for s in sub_summary) - min(s[2] for s in sub_summary)
        spread_v50 = max(s[4] for s in sub_summary) - min(s[4] for s in sub_summary)
        spread_d90 = max(s[3] for s in sub_summary) - min(s[3] for s in sub_summary)
        spread_v90 = max(s[5] for s in sub_summary) - min(s[5] for s in sub_summary)
        print(f"\n  Spread across sub-periods:")
        print(f"    distance/ATR median  : {spread_d50:.3f}")
        print(f"    distance/ATR 90th    : {spread_d90:.3f}")
        print(f"    velocity     median  : {spread_v50:.3f}")
        print(f"    velocity     90th    : {spread_v90:.3f}")
        print(f"  (1H reference for comparison: dist median spread = 0.036)")
    plt.show()

    # ── Recommendations ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  RECOMMENDED 5M PRE-REGISTERED VALUES")
    print(f"{'='*72}")
    print(f"  SL_ATR_MULT_5M       = {median_dam:.3f}")
    print(f"  MIN_PROBE_DIST_ATR_5M = ?       (pick from histogram — natural break or 10-25th pct)")
    print(f"  VELOCITY_THRESHOLD_5M = ?       (pick from histogram — high tail / 85-95th pct)")
    print(f"")
    print(f"  Heuristic ranges:")
    print(f"    Min probe dist  : 10th–25th pct → "
          f"{np.percentile(dam, 10):.3f}–{np.percentile(dam, 25):.3f}")
    print(f"    Velocity thresh : 85th–95th pct → "
          f"{np.percentile(vs_finite, 85):.3f}–{np.percentile(vs_finite, 95):.3f}")
    print(f"")
    print(f"  REFERENCE — original 1H locked values:")
    print(f"    SL_ATR_MULT_TPO    = 1.578")
    print(f"    MIN_PROBE_DIST_ATR = 0.40")
    print(f"    VELOCITY_THRESHOLD = 2.21")
    print(f"{'='*72}\n")

    print("  Look at probe_distributions_5m.png and stability_check_5m.png.")
    print("  Pick MIN_PROBE_DIST_ATR_5M and VELOCITY_THRESHOLD_5M from the")
    print("  distributions, send all three values, and we lock them in")
    print("  Config under new keys (preserving original 1H locks for the")
    print("  paper's transparency requirements).")


if __name__ == "__main__":
    main()
