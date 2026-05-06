# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — PRE-REGISTRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per IB_Auction_Fade_Framework.pdf Section 3.6 and TRANSITION_PROMPT_V2 Phase 2.
#
# Runs ONCE on the full 2015–2026 dataset. Produces the empirical distributions
# from which three locked parameters are pre-registered before the confirmation
# backtest runs:
#
#   1. SL_ATR_MULT       — auto-set as the MEDIAN of (probe distance / ATR)
#   2. MIN_PROBE_DIST_ATR — you pick by eye from the distance distribution
#   3. VELOCITY_THRESHOLD — you pick by eye from the velocity-score distribution
#
# After this script runs, write all three values down. They become fixed inputs
# to Phase 3 (state-machine rebuild) and Phase 4 (confirmation backtest). They
# do not change after results are observed. That is the pre-registration rule.
#
# Usage:
#     python pre_registration_analysis.py
#
# Outputs (saved to results\pre_registration\):
#     probe_distribution.csv   — full event-level data
#     probe_distributions.png  — 4-panel histogram figure
#
# Runtime: 1–5 minutes depending on machine. Engine processes ~67k 1H bars
# end-to-end; profile + VA recomputation per bar is the dominant cost.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy import Config, load_and_prepare_data
from tpo_engine import TPOEngine


# ── Analysis range (full dataset per framework; do not narrow) ────────────────
ANALYSIS_START = "2015-01-01"
ANALYSIS_END   = "2026-12-31"

OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\pre_registration"


# ═══════════════════════════════════════════════════════════════════════════════
# ATR HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df_1h, period=14):
    """
    Standard 14-period ATR on hourly bars.
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    h = df_1h['high'].values
    l = df_1h['low'].values
    c = df_1h['close'].values
    pc = np.concatenate([[c[0]], c[:-1]])  # prev close, first row uses self
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    return atr


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  PRE-REGISTRATION ANALYSIS")
    print(f"  {ANALYSIS_START}  →  {ANALYSIS_END}")
    print(f"{'='*70}\n")

    # ── Load full range data ──────────────────────────────────────────────────
    saved_range = (Config.BACKTEST_START, Config.BACKTEST_END)
    Config.BACKTEST_START = ANALYSIS_START
    Config.BACKTEST_END   = ANALYSIS_END
    try:
        df_1h, df_1m = load_and_prepare_data()
    finally:
        Config.BACKTEST_START, Config.BACKTEST_END = saved_range

    print(f"\nLoaded {len(df_1h):,} 1H bars and {len(df_1m):,} 1M bars")

    # ── Precompute ATR(14), bar range, median volume ──────────────────────────
    df_1h = df_1h.copy().reset_index(drop=True)
    df_1h['atr']   = compute_atr(df_1h, period=Config.ATR_PERIOD)
    df_1h['range'] = df_1h['high'] - df_1h['low']

    median_volume = float(df_1h['volume'].median())
    print(f"Median hourly volume (global): {median_volume:,.0f}")
    print(f"NOTE: production state machine should use ROLLING median for live use.")
    print(f"      Global median is fine for this offline analysis only.\n")

    # ── Build 1M index for fast hourly lookup ─────────────────────────────────
    df_1m = df_1m.copy()
    df_1m['hour_bucket'] = df_1m['datetime'].dt.floor('h')
    print("Building 1M index...")
    m_index = {hour: g.reset_index(drop=True)
               for hour, g in df_1m.groupby('hour_bucket')}
    print(f"  {len(m_index):,} hourly buckets indexed")

    # ── Walk through 1H bars; collect probe events ────────────────────────────
    print(f"\nRunning TPO engine over full range...")
    engine = TPOEngine()
    probes = []
    t0 = time.time()
    n_bars = len(df_1h)

    for i, row in df_1h.iterrows():
        bar_minutes = m_index.get(row['datetime'])
        engine.add_bar(row, bar_minutes)

        # Skip until weekly profile has data
        if engine.weekly_vah is None or engine.weekly_val is None:
            continue

        atr_v = row['atr']
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue

        close_v = row['close']
        if close_v > engine.weekly_vah:
            direction     = 'above'
            probe_extreme = row['high']
            distance      = probe_extreme - engine.weekly_vah
        elif close_v < engine.weekly_val:
            direction     = 'below'
            probe_extreme = row['low']
            distance      = engine.weekly_val - probe_extreme
        else:
            continue  # not a probe event

        if distance < 0:
            distance = 0.0  # guard against rounding artifacts

        distance_atr_mult = distance / atr_v

        bar_vol = float(row['volume']) if np.isfinite(row['volume']) else 0.0
        if bar_vol > 0:
            velocity_score = (row['range'] / atr_v) / (bar_vol / median_volume)
        else:
            velocity_score = float('inf')

        # Hours into the week (since last weekly reset)
        if engine.weekly_reset_dt is not None:
            hrs_into_wk = (row['datetime'] - engine.weekly_reset_dt).total_seconds() / 3600.0
        else:
            hrs_into_wk = np.nan

        probes.append({
            'datetime'         : row['datetime'],
            'direction'        : direction,
            'close'            : float(close_v),
            'probe_extreme'    : float(probe_extreme),
            'vah'              : float(engine.weekly_vah),
            'val'              : float(engine.weekly_val),
            'poc'              : float(engine.weekly_poc),
            'atr'              : float(atr_v),
            'bar_range'        : float(row['range']),
            'bar_volume'       : bar_vol,
            'distance'         : float(distance),
            'distance_atr_mult': float(distance_atr_mult),
            'velocity_score'   : float(velocity_score),
            'hrs_into_week'    : float(hrs_into_wk),
        })

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:>6,}/{n_bars:,}]  probes: {len(probes):>5,}  elapsed: {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_bars:,} bars in {elapsed:.0f}s. Collected {len(probes):,} probe events.")

    df_probes = pd.DataFrame(probes)

    # ── Save raw event data ───────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "probe_distribution.csv")
    df_probes.to_csv(csv_path, index=False)
    print(f"\nProbe events saved: {csv_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_total = len(df_probes)
    if n_total == 0:
        print("\n[WARN] No probe events found — engine produced no closes outside developing VA.")
        return

    n_above = int((df_probes['direction'] == 'above').sum())
    n_below = int((df_probes['direction'] == 'below').sum())

    print(f"\n{'─'*70}")
    print(f"  PROBE DISTRIBUTION SUMMARY")
    print(f"{'─'*70}")
    print(f"  Total probes      : {n_total:,}")
    print(f"  Above weekly VAH  : {n_above:,}  ({n_above/n_total*100:.1f}%)")
    print(f"  Below weekly VAL  : {n_below:,}  ({n_below/n_total*100:.1f}%)")

    # Distance / ATR percentiles
    dam = df_probes['distance_atr_mult'].values
    median_dam = float(np.median(dam))
    print(f"\n  Distance / ATR(14) percentiles:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = float(np.percentile(dam, pct))
        marker = "  <-- median (= recommended SL_ATR_MULT)" if pct == 50 else ""
        print(f"    {pct:>3d}th pct : {v:>7.3f}{marker}")

    # Velocity score percentiles (drop inf)
    vs_all = df_probes['velocity_score'].values
    vs_finite = vs_all[np.isfinite(vs_all)]
    median_vs = float(np.median(vs_finite))
    n_inf = int(np.sum(~np.isfinite(vs_all)))
    print(f"\n  Velocity score percentiles  (n_finite={len(vs_finite):,}, n_inf={n_inf}):")
    for pct in [1, 5, 10, 25, 50, 75, 85, 90, 95, 99]:
        v = float(np.percentile(vs_finite, pct))
        print(f"    {pct:>3d}th pct : {v:>7.3f}")

    # ── Plot the four diagnostic panels ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor='#131722')
    for ax in axes.flat:
        ax.set_facecolor('#1e222d')
        for spine in ax.spines.values():
            spine.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

    # Top-left: distance/ATR linear histogram (clipped to 99th pct for visibility)
    dam_top = float(np.percentile(dam, 99))
    dam_clip = dam[dam <= dam_top]
    axes[0, 0].hist(dam_clip, bins=100, color='#5dade2', edgecolor='#1e222d', alpha=0.85)
    axes[0, 0].axvline(median_dam, color='#f1c40f', linestyle='--', linewidth=1.8,
                       label=f'Median = {median_dam:.3f}')
    axes[0, 0].set_title(f'Probe Distance / ATR(14)  — n={len(dam):,}, clipped at 99th pct')
    axes[0, 0].set_xlabel('distance / ATR')
    axes[0, 0].set_ylabel('count')
    axes[0, 0].legend(facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='#d1d4dc', fontsize=9)

    # Top-right: distance/ATR CDF
    sorted_dam = np.sort(dam)
    cdf = np.arange(1, len(sorted_dam) + 1) / len(sorted_dam)
    axes[0, 1].plot(sorted_dam, cdf, color='#5dade2', linewidth=1.4)
    axes[0, 1].axvline(median_dam, color='#f1c40f', linestyle='--', linewidth=1.5,
                       label=f'50th pct = {median_dam:.3f}')
    for pct in [10, 25, 75, 90]:
        v = float(np.percentile(dam, pct))
        axes[0, 1].axvline(v, color='#5dade2', linestyle=':', linewidth=0.8, alpha=0.7)
        axes[0, 1].text(v, 0.05, f'{pct}th\n{v:.2f}', color='#d1d4dc', fontsize=7,
                        ha='center', alpha=0.8)
    axes[0, 1].set_title('Probe Distance / ATR(14) — CDF')
    axes[0, 1].set_xlabel('distance / ATR')
    axes[0, 1].set_ylabel('cumulative fraction')
    axes[0, 1].set_xlim(0, dam_top)
    axes[0, 1].legend(facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='#d1d4dc', fontsize=9)

    # Bottom-left: velocity score histogram (clipped)
    vs_top = float(np.percentile(vs_finite, 99))
    vs_clip = vs_finite[vs_finite <= vs_top]
    axes[1, 0].hist(vs_clip, bins=100, color='#bb6bd9', edgecolor='#1e222d', alpha=0.85)
    axes[1, 0].axvline(median_vs, color='#f1c40f', linestyle='--', linewidth=1.8,
                       label=f'Median = {median_vs:.3f}')
    axes[1, 0].set_title(f'Velocity Score  — n={len(vs_finite):,}, clipped at 99th pct')
    axes[1, 0].set_xlabel('velocity_score  =  (range/ATR) / (vol/median_vol)')
    axes[1, 0].set_ylabel('count')
    axes[1, 0].legend(facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='#d1d4dc', fontsize=9)

    # Bottom-right: velocity CDF
    sorted_vs = np.sort(vs_finite)
    vs_cdf = np.arange(1, len(sorted_vs) + 1) / len(sorted_vs)
    axes[1, 1].plot(sorted_vs, vs_cdf, color='#bb6bd9', linewidth=1.4)
    axes[1, 1].axvline(median_vs, color='#f1c40f', linestyle='--', linewidth=1.5,
                       label=f'50th pct = {median_vs:.3f}')
    for pct in [75, 85, 90, 95]:
        v = float(np.percentile(vs_finite, pct))
        axes[1, 1].axvline(v, color='#bb6bd9', linestyle=':', linewidth=0.8, alpha=0.7)
        axes[1, 1].text(v, 0.05, f'{pct}th\n{v:.2f}', color='#d1d4dc', fontsize=7,
                        ha='center', alpha=0.8)
    axes[1, 1].set_title('Velocity Score — CDF')
    axes[1, 1].set_xlabel('velocity_score')
    axes[1, 1].set_ylabel('cumulative fraction')
    axes[1, 1].set_xlim(0, vs_top)
    axes[1, 1].legend(facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='#d1d4dc', fontsize=9)

    fig.suptitle(
        f'Pre-Registration Distributions  —  {n_total:,} probes  —  '
        f'{ANALYSIS_START} to {ANALYSIS_END}',
        color='#d1d4dc', fontsize=13,
    )
    fig.tight_layout()

    fig_path = os.path.join(OUT_DIR, "probe_distributions.png")
    fig.savefig(fig_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    print(f"\nDistributions plot saved: {fig_path}")
    plt.show()

    # ═════════════════════════════════════════════════════════════════════════
    # SUB-PERIOD STABILITY CHECK
    # ═════════════════════════════════════════════════════════════════════════
    # Splits probes into ~3-year sub-periods and overlays their CDFs.
    # If lines overlap closely → distributions are stable across regimes,
    # locked parameters are valid forever.
    # If lines diverge → distributions are drifting, you should pre-register
    # a recalibration schedule instead of single fixed values.
    df_probes['year'] = pd.to_datetime(df_probes['datetime']).dt.year
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
        for spine in ax.spines.values():
            spine.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

    print(f"\n{'─'*70}")
    print(f"  SUB-PERIOD STABILITY CHECK")
    print(f"{'─'*70}")
    print(f"  {'Period':<12}{'n_probes':>10}{'p50_dist':>12}{'p90_dist':>12}{'p50_vel':>12}{'p90_vel':>12}")

    sub_summary = []
    for (label, (y0, y1)), color in zip(sub_periods, sub_colors):
        sub = df_probes[(df_probes['year'] >= y0) & (df_probes['year'] <= y1)]
        if len(sub) < 50:
            print(f"  {label:<12}{len(sub):>10}    (skipping — too few probes)")
            continue

        sub_dam = sub['distance_atr_mult'].values
        sub_vs  = sub['velocity_score'].values
        sub_vs  = sub_vs[np.isfinite(sub_vs)]

        # Distance CDF
        s_dam = np.sort(sub_dam)
        c_dam = np.arange(1, len(s_dam) + 1) / len(s_dam)
        axes2[0].plot(s_dam, c_dam, color=color, linewidth=1.4,
                      label=f'{label}  (n={len(sub):,})')

        # Velocity CDF
        s_vs = np.sort(sub_vs)
        c_vs = np.arange(1, len(s_vs) + 1) / len(s_vs)
        axes2[1].plot(s_vs, c_vs, color=color, linewidth=1.4,
                      label=f'{label}  (n={len(sub):,})')

        p50_d = float(np.percentile(sub_dam, 50))
        p90_d = float(np.percentile(sub_dam, 90))
        p50_v = float(np.percentile(sub_vs,  50))
        p90_v = float(np.percentile(sub_vs,  90))
        sub_summary.append((label, len(sub), p50_d, p90_d, p50_v, p90_v))
        print(f"  {label:<12}{len(sub):>10,}{p50_d:>12.3f}{p90_d:>12.3f}{p50_v:>12.3f}{p90_v:>12.3f}")

    axes2[0].set_title('Distance / ATR — CDF by sub-period')
    axes2[0].set_xlabel('distance / ATR')
    axes2[0].set_ylabel('cumulative fraction')
    axes2[0].set_xlim(0, dam_top)
    axes2[0].legend(facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='#d1d4dc', fontsize=8)

    axes2[1].set_title('Velocity Score — CDF by sub-period')
    axes2[1].set_xlabel('velocity_score')
    axes2[1].set_ylabel('cumulative fraction')
    axes2[1].set_xlim(0, vs_top)
    axes2[1].legend(facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='#d1d4dc', fontsize=8)

    fig2.suptitle('Sub-Period Stability — overlap = stable, divergence = drift',
                  color='#d1d4dc', fontsize=12)
    fig2.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "stability_check.png")
    fig2.savefig(fig2_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    print(f"\n  Stability plot saved: {fig2_path}")

    # Quick numerical drift summary: max spread of medians across periods
    if len(sub_summary) >= 2:
        p50_d_spread = max(s[2] for s in sub_summary) - min(s[2] for s in sub_summary)
        p50_v_spread = max(s[4] for s in sub_summary) - min(s[4] for s in sub_summary)
        p90_d_spread = max(s[3] for s in sub_summary) - min(s[3] for s in sub_summary)
        p90_v_spread = max(s[5] for s in sub_summary) - min(s[5] for s in sub_summary)
        print(f"\n  Median spread across sub-periods (smaller = more stable):")
        print(f"    distance/ATR median  : {p50_d_spread:.3f}")
        print(f"    distance/ATR 90th pct: {p90_d_spread:.3f}")
        print(f"    velocity     median  : {p50_v_spread:.3f}")
        print(f"    velocity     90th pct: {p90_v_spread:.3f}")
        print(f"\n  Heuristic interpretation:")
        print(f"    distance/ATR median spread < 0.10  → stable enough to lock")
        print(f"    distance/ATR median spread > 0.20  → consider recalibration schedule")
    plt.show()

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RECOMMENDED PRE-REGISTERED VALUES")
    print(f"{'='*70}")
    print(f"  SL_ATR_MULT          = {median_dam:.3f}  (median of distance distribution — auto)")
    print(f"  MIN_PROBE_DIST_ATR   = ?       (pick from histogram — natural break point)")
    print(f"  VELOCITY_THRESHOLD   = ?       (pick from histogram — high tail = spikes)")
    print(f"")
    print(f"  Heuristics if no obvious break is visible:")
    print(f"    Min probe dist  : 10th–25th percentile of distance distribution")
    print(f"                      → range here: "
          f"{np.percentile(dam, 10):.3f}–{np.percentile(dam, 25):.3f}")
    print(f"    Velocity thresh : 85th–95th percentile of velocity distribution")
    print(f"                      → range here: "
          f"{np.percentile(vs_finite, 85):.3f}–{np.percentile(vs_finite, 95):.3f}")
    print(f"{'='*70}\n")

    print("Look at the 4-panel figure. For each distribution, ask:")
    print("  - Is there a visible elbow / discontinuity / multi-modal shape?")
    print("  - If yes, pick the value at that break.")
    print("  - If no, fall back to the heuristic percentile range above.")
    print()
    print("Once you've picked all three values, send them to me and we lock them in")
    print("Config — then move on to Phase 3 (state machine rebuild).")


if __name__ == "__main__":
    main()
