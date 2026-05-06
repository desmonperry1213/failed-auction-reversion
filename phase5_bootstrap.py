# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5a — BOOTSTRAP STATISTICAL TESTING (ES single-instrument)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per IB_Auction_Fade_Framework.pdf Section 4.
#
# Loads trades.csv from Phase 4 and runs:
#   1. Primary bootstrap test on full sample        (framework Section 4.1)
#   2. ATR-tertile regime-conditional sub-tests     (framework Section 4.2)
#   3. Sub-period stability sub-tests               (added based on equity-curve shape)
#   4. Long/short directional bootstraps            (added based on observed asymmetry)
#   5. Maximum drawdown bootstrap distribution      (framework Section 4.3)
#
# Method (Section 4.1):
#   For each test:
#     - Compute observed avg-R = mean(r_1..r_N)
#     - 10,000 bootstrap resamples drawn with replacement, size N
#     - For each resample, compute avg-R → bootstrap distribution
#     - p-value (one-tailed) = fraction of (resample_R - obs_R) ≥ obs_R
#       i.e., shift-to-null procedure per framework Section 4.1
#     - 95% CI = [2.5th, 97.5th] percentile of unshifted bootstrap distribution
#
# Cross-instrument replication and Fisher's combined test (framework Section 4.4)
# is Phase 5b, deferred until NQ/GC/ZN data is acquired.
#
# Outputs (saved to results\bootstrap\):
#   bootstrap_summary.txt        — Table-1-style terminal report
#   bootstrap_full_atr.png       — full sample + ATR tertiles (4 panels)
#   bootstrap_subperiod_dir.png  — 4 sub-periods + long/short (6 panels)
#   drawdown_distribution.png    — bootstrap max-DD distribution
# ═══════════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────────────
TRADES_CSV = r"C:\Trading\InitialBalanceAuctionFade\results\confirmation_backtest\trades.csv"
OUT_DIR    = r"C:\Trading\InitialBalanceAuctionFade\results\bootstrap"

N_BOOTSTRAP = 10_000
SEED        = 42  # deterministic reproducibility


# ═══════════════════════════════════════════════════════════════════════════════
# CORE BOOTSTRAP MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_avg_r(r_values, n_resamples=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap distribution of avg-R per framework Section 4.1.

    Returns dict with:
        n          : sample size
        obs        : observed avg-R
        resamples  : array of n_resamples bootstrap means
        ci_low     : 2.5th percentile
        ci_high    : 97.5th percentile
        p_value    : one-tailed bootstrap p-value (shift-to-null)
    """
    r = np.asarray(r_values, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return {
            'n': 0, 'obs': 0.0, 'resamples': np.array([]),
            'ci_low': 0.0, 'ci_high': 0.0, 'p_value': float('nan'),
        }

    rng = np.random.default_rng(seed)
    obs = float(np.mean(r))

    # Vectorized bootstrap: draw n_resamples × n indices, take row means
    idx       = rng.integers(0, n, size=(n_resamples, n))
    resamples = r[idx].mean(axis=1)

    ci_low  = float(np.percentile(resamples, 2.5))
    ci_high = float(np.percentile(resamples, 97.5))

    # Shift-to-null p-value
    shifted = resamples - obs
    p_value = float(np.mean(shifted >= obs))

    return {
        'n': n, 'obs': obs, 'resamples': resamples,
        'ci_low': ci_low, 'ci_high': ci_high, 'p_value': p_value,
    }


def bootstrap_max_drawdown(r_values, n_resamples=N_BOOTSTRAP, seed=SEED):
    """
    For each resample (with replacement), compute the cumulative R-curve
    in the resample's order and record the maximum drawdown in R units.

    Returns dict with median, 95th pct, 99th pct of drawdown distribution.
    """
    r = np.asarray(r_values, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return {
            'n': 0, 'observed_dd': 0.0, 'median_dd': 0.0,
            'p95_dd': 0.0, 'p99_dd': 0.0, 'distribution': np.array([]),
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resamples = r[idx]                              # (n_resamples, n)
    cum       = np.cumsum(resamples, axis=1)        # cumulative R curves
    peak      = np.maximum.accumulate(cum, axis=1)  # running peaks
    dd        = peak - cum                          # drawdowns from peak
    max_dd    = dd.max(axis=1)                      # per-resample max DD

    # Observed DD on the actual sequence (just for reference)
    cum_obs   = np.cumsum(r)
    peak_obs  = np.maximum.accumulate(cum_obs)
    dd_obs    = (peak_obs - cum_obs).max()

    return {
        'n'           : n,
        'observed_dd' : float(dd_obs),
        'median_dd'   : float(np.median(max_dd)),
        'p95_dd'      : float(np.percentile(max_dd, 95)),
        'p99_dd'      : float(np.percentile(max_dd, 99)),
        'distribution': max_dd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_test_row(label, result):
    """One Table-1 row in fixed-width prose."""
    if result['n'] == 0:
        return f"  {label:<26}{'(no trades)':<60}"
    sig = "*" if result['p_value'] < 0.05 else " "
    return (f"  {label:<26}{result['n']:>5}  "
            f"{result['obs']:>+8.4f}  "
            f"[{result['ci_low']:>+7.4f}, {result['ci_high']:>+7.4f}]  "
            f"{result['p_value']:>7.4f}{sig}")


def plot_bootstrap_grid(results_dict, fig_title, out_path):
    """
    results_dict: {label: bootstrap_result_dict}
    Plots bootstrap distributions as histograms with observed value marked.
    """
    n_panels = len(results_dict)
    cols     = 2
    rows     = (n_panels + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3 * rows), facecolor='#131722')
    axes = np.array(axes).flatten() if n_panels > 1 else [axes]

    for i, (label, res) in enumerate(results_dict.items()):
        ax = axes[i]
        ax.set_facecolor('#1e222d')
        for sp in ax.spines.values():
            sp.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

        if res['n'] == 0:
            ax.text(0.5, 0.5, f"{label}\n(no trades)", color='#d1d4dc',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        bins = ax.hist(res['resamples'], bins=60, color='#5dade2',
                       edgecolor='#1e222d', alpha=0.85)
        ax.axvline(res['obs'],     color='#f1c40f', linestyle='-',  linewidth=1.8,
                   label=f"obs = {res['obs']:+.4f}")
        ax.axvline(res['ci_low'],  color='#bb6bd9', linestyle='--', linewidth=1.0,
                   label=f"CI low  = {res['ci_low']:+.4f}")
        ax.axvline(res['ci_high'], color='#bb6bd9', linestyle='--', linewidth=1.0,
                   label=f"CI high = {res['ci_high']:+.4f}")
        ax.axvline(0, color='#d1d4dc', linewidth=0.6, alpha=0.4)

        sig = "*" if res['p_value'] < 0.05 else ""
        ax.set_title(f"{label}  (n={res['n']})  p={res['p_value']:.4f}{sig}",
                     fontsize=10)
        ax.set_xlabel('avg-R'); ax.set_ylabel('count')
        ax.legend(facecolor='#1e222d', edgecolor='#2a2e39',
                  labelcolor='#d1d4dc', fontsize=7)

    # Hide any unused panels
    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    fig.suptitle(fig_title, color='#d1d4dc', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*72}")
    print("  PHASE 5a — BOOTSTRAP STATISTICAL TESTING (ES single-instrument)")
    print(f"{'='*72}\n")

    if not os.path.exists(TRADES_CSV):
        print(f"  ERROR: {TRADES_CSV} not found. Run run_tpo_confirmation.py first.")
        return

    df = pd.read_csv(TRADES_CSV)
    df['entry_dt'] = pd.to_datetime(df['entry_dt'], utc=True, errors='coerce')
    print(f"  Loaded {len(df):,} trades from {TRADES_CSV}")
    print(f"  Bootstrap resamples per test: {N_BOOTSTRAP:,}")
    print(f"  Random seed (deterministic):  {SEED}\n")

    results = {}  # all bootstrap results, keyed by label

    # ── 1. Primary test ──────────────────────────────────────────────────────
    print("  Running primary bootstrap (full sample)...")
    results['Full sample'] = bootstrap_avg_r(df['r_multiple'])

    # ── 2. ATR tertile sub-tests (framework Section 4.2) ─────────────────────
    print("  Running ATR-tertile sub-tests...")
    p33 = float(np.percentile(df['atr_at_entry'], 33))
    p67 = float(np.percentile(df['atr_at_entry'], 67))
    low_atr  = df[df['atr_at_entry'] <  p33]
    mid_atr  = df[(df['atr_at_entry'] >= p33) & (df['atr_at_entry'] <= p67)]
    high_atr = df[df['atr_at_entry'] >  p67]
    results['Low ATR (<P33)']         = bootstrap_avg_r(low_atr['r_multiple'])
    results['Medium ATR (P33-P67)']   = bootstrap_avg_r(mid_atr['r_multiple'])
    results['High ATR (>P67)']        = bootstrap_avg_r(high_atr['r_multiple'])

    # ── 3. Sub-period sub-tests (added) ──────────────────────────────────────
    print("  Running sub-period sub-tests...")
    df['year'] = df['entry_dt'].dt.year
    sub_periods = [
        ('2015-2017', (2015, 2017)),
        ('2018-2020', (2018, 2020)),
        ('2021-2023', (2021, 2023)),
        ('2024-2026', (2024, 2026)),
    ]
    for label, (y0, y1) in sub_periods:
        sub = df[(df['year'] >= y0) & (df['year'] <= y1)]
        results[label] = bootstrap_avg_r(sub['r_multiple'])

    # ── 4. Long/short sub-tests (added) ──────────────────────────────────────
    print("  Running long/short sub-tests...")
    longs  = df[df['direction'] == 'long']
    shorts = df[df['direction'] == 'short']
    results['Longs only']  = bootstrap_avg_r(longs['r_multiple'])
    results['Shorts only'] = bootstrap_avg_r(shorts['r_multiple'])

    # ── 5. Maximum drawdown bootstrap (framework Section 4.3) ────────────────
    print("  Running maximum-drawdown bootstrap...")
    dd = bootstrap_max_drawdown(df['r_multiple'])

    # ═════════════════════════════════════════════════════════════════════════
    # TABLE 1
    # ═════════════════════════════════════════════════════════════════════════
    lines = []
    lines.append("="*72)
    lines.append("TABLE 1 — Bootstrap Statistical Test Results (ES, 2015-01 to 2026-05)")
    lines.append("="*72)
    lines.append(f"  Resamples per test : {N_BOOTSTRAP:,}")
    lines.append(f"  Random seed        : {SEED}")
    lines.append(f"  Method             : One-tailed shift-to-null bootstrap (Section 4.1)")
    lines.append(f"  CI                 : [2.5th, 97.5th] percentile of bootstrap distribution")
    lines.append(f"  Significance       : p < 0.05  (marked with * after p-value)")
    lines.append("")
    lines.append(f"  {'Test':<26}{'n':>5}  {'avg-R':>8}  "
                 f"{'95% CI':<22}  {'p-value':>8}")
    lines.append("  " + "-"*70)

    primary_label = 'Full sample'
    lines.append(fmt_test_row(primary_label, results[primary_label]))
    lines.append("")
    lines.append("  ATR regime sub-tests:")
    for k in ['Low ATR (<P33)', 'Medium ATR (P33-P67)', 'High ATR (>P67)']:
        lines.append(fmt_test_row(k, results[k]))
    lines.append("")
    lines.append("  Sub-period sub-tests:")
    for label, _ in sub_periods:
        lines.append(fmt_test_row(label, results[label]))
    lines.append("")
    lines.append("  Direction sub-tests:")
    for k in ['Longs only', 'Shorts only']:
        lines.append(fmt_test_row(k, results[k]))
    lines.append("")
    lines.append("="*72)
    lines.append("TABLE 2 — Maximum Drawdown Bootstrap Distribution (R units)")
    lines.append("="*72)
    lines.append(f"  Sample size           : n={dd['n']}")
    lines.append(f"  Observed max DD       : {dd['observed_dd']:7.3f}R")
    lines.append(f"  Bootstrap median DD   : {dd['median_dd']:7.3f}R")
    lines.append(f"  Bootstrap 95th pct DD : {dd['p95_dd']:7.3f}R   <-- realistic worst case")
    lines.append(f"  Bootstrap 99th pct DD : {dd['p99_dd']:7.3f}R")
    lines.append("")
    lines.append("="*72)
    lines.append("INTERPRETATION GUIDE")
    lines.append("="*72)
    lines.append("  * p-value < 0.05  : reject null at 95% confidence; evidence of edge")
    lines.append("  * 95% CI excludes 0: same finding viewed from the CI side")
    lines.append("  * For the regime sub-tests, a positive finding in some regimes but not")
    lines.append("    others is a CONDITIONAL EDGE — academically valid and informative")
    lines.append("    for deployment (framework Section 4.2).")
    lines.append("  * For the sub-period sub-tests, divergence indicates either regime")
    lines.append("    drift or strategy decay through arbitrage (cross-instrument test")
    lines.append("    in Phase 5b is needed to disambiguate).")
    lines.append("  * 95th percentile drawdown is the realistic worst case to plan for.")
    lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)

    # Save summary
    out_summary = os.path.join(OUT_DIR, "bootstrap_summary.txt")
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n  Saved: {out_summary}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n  Generating figures...")

    # Figure 1: Full + ATR tertiles
    fig1_results = {
        'Full sample'             : results['Full sample'],
        'Low ATR (<P33)'          : results['Low ATR (<P33)'],
        'Medium ATR (P33-P67)'    : results['Medium ATR (P33-P67)'],
        'High ATR (>P67)'         : results['High ATR (>P67)'],
    }
    plot_bootstrap_grid(
        fig1_results,
        'Phase 5a — Bootstrap Distributions: Full Sample + ATR Tertiles',
        os.path.join(OUT_DIR, "bootstrap_full_atr.png"),
    )

    # Figure 2: Sub-periods + direction
    fig2_results = {
        '2015-2017'   : results['2015-2017'],
        '2018-2020'   : results['2018-2020'],
        '2021-2023'   : results['2021-2023'],
        '2024-2026'   : results['2024-2026'],
        'Longs only'  : results['Longs only'],
        'Shorts only' : results['Shorts only'],
    }
    plot_bootstrap_grid(
        fig2_results,
        'Phase 5a — Bootstrap Distributions: Sub-Periods + Direction',
        os.path.join(OUT_DIR, "bootstrap_subperiod_dir.png"),
    )

    # Figure 3: Drawdown distribution
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#131722')
    ax.set_facecolor('#1e222d')
    for sp in ax.spines.values():
        sp.set_color('#2a2e39')
    ax.tick_params(colors='#d1d4dc', labelsize=8)
    ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
    ax.title.set_color('#d1d4dc')
    ax.xaxis.label.set_color('#d1d4dc')
    ax.yaxis.label.set_color('#d1d4dc')

    ax.hist(dd['distribution'], bins=80, color='#ef5350',
            edgecolor='#1e222d', alpha=0.85)
    ax.axvline(dd['observed_dd'], color='#f1c40f', linestyle='-', linewidth=1.8,
               label=f"observed = {dd['observed_dd']:.2f}R")
    ax.axvline(dd['median_dd'],  color='#26a69a', linestyle='--', linewidth=1.4,
               label=f"median = {dd['median_dd']:.2f}R")
    ax.axvline(dd['p95_dd'],     color='#bb6bd9', linestyle='--', linewidth=1.4,
               label=f"95th pct = {dd['p95_dd']:.2f}R")
    ax.axvline(dd['p99_dd'],     color='#bb6bd9', linestyle=':',  linewidth=1.4,
               label=f"99th pct = {dd['p99_dd']:.2f}R")
    ax.set_title(f"Maximum Drawdown Bootstrap Distribution  "
                 f"(n={dd['n']}, {N_BOOTSTRAP:,} resamples)", fontsize=12)
    ax.set_xlabel('Max drawdown (R units)')
    ax.set_ylabel('count')
    ax.legend(facecolor='#1e222d', edgecolor='#2a2e39',
              labelcolor='#d1d4dc', fontsize=9)
    fig.tight_layout()
    dd_path = os.path.join(OUT_DIR, "drawdown_distribution.png")
    fig.savefig(dd_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    plt.show()
    print(f"  Saved: {dd_path}")

    print(f"\n{'='*72}")
    print("  Phase 5a complete.")
    print(f"{'='*72}")
    print()
    print("  Next step decision tree:")
    print("    - If primary test p < 0.05            → strong, proceed to cross-instrument")
    print("    - If primary p > 0.05 but conditional →")
    print("        e.g., one ATR tertile or sub-period is significant alone — this is")
    print("        a CONDITIONAL EDGE and informs Phase 5b cross-instrument design")
    print("    - If primary AND all sub-tests p > 0.05 → null cannot be rejected on ES;")
    print("        cross-instrument pooling is the only path to a publishable finding")
    print()


if __name__ == "__main__":
    main()
