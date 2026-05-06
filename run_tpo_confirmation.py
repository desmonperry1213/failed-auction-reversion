# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — CONFIRMATION BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the TPO state machine ONCE on the full 2015–2026 ES dataset using the
# pre-registered locked parameters. Records all trades. Produces a summary
# report. NO PARAMETER CHANGES PERMITTED AFTER THIS POINT regardless of the
# results.
#
# This is the moment of truth. Per the framework discipline: whatever this
# script produces is the result. If avg-R is positive, we move to Phase 5
# (bootstrap statistical testing). If avg-R is negative, that is also a
# valid scientific result and we report it as such — the null hypothesis
# was not rejected.
#
# Outputs (saved to results\confirmation_backtest\):
#     trades.csv           — one row per closed trade with full setup metrics
#     summary.txt          — terminal-style summary
#     equity_curve.png     — cumulative R-multiple and dollar P&L over time
#
# Runtime expectation: 2–5 minutes (slightly slower than pre-registration
# analysis because of the per-bar state-machine logic).
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy import Config, load_and_prepare_data
from tpo_state_machine import TPOStateMachine, trades_to_dataframe


CONFIRMATION_START = "2015-01-01"
CONFIRMATION_END   = "2026-12-31"
OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\confirmation_backtest"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  PHASE 4 — CONFIRMATION BACKTEST")
    print(f"  {CONFIRMATION_START}  →  {CONFIRMATION_END}")
    print(f"{'='*70}")
    print(f"\n  PRE-REGISTERED PARAMETERS (locked 2026-05-05):")
    print(f"    SL_ATR_MULT_TPO      = {Config.SL_ATR_MULT_TPO}")
    print(f"    MIN_PROBE_DIST_ATR   = {Config.MIN_PROBE_DIST_ATR}")
    print(f"    VELOCITY_THRESHOLD   = {Config.VELOCITY_THRESHOLD}")
    print(f"    VA_THRESHOLD         = {Config.VA_THRESHOLD}")
    print(f"    SL_ATR_PERIOD        = {Config.ATR_PERIOD}")
    print(f"\n  These do NOT change based on results below.\n")

    # Enforce OHLC-decomposition delta per framework — overrides Config
    saved_delta_method = getattr(Config, 'DELTA_METHOD', 'close_vs_open')
    Config.DELTA_METHOD = 'ohlc_decomposition'
    print(f"  Delta method (forced) = {Config.DELTA_METHOD}")
    if saved_delta_method != 'ohlc_decomposition':
        print(f"  (Config.DELTA_METHOD was {saved_delta_method!r}, overridden for this run)")
    print()

    # Load full dataset
    saved_range = (Config.BACKTEST_START, Config.BACKTEST_END)
    Config.BACKTEST_START = CONFIRMATION_START
    Config.BACKTEST_END   = CONFIRMATION_END
    try:
        df_1h, df_1m = load_and_prepare_data()
    finally:
        Config.BACKTEST_START, Config.BACKTEST_END = saved_range

    print(f"\nDataset: {len(df_1h):,} 1H bars, {len(df_1m):,} 1M bars\n")

    # Run state machine
    print("Running TPO state machine over full range...")
    t0 = time.time()
    sm = TPOStateMachine(df_1h, df_1m, contracts=1, verbose=False)
    trades = sm.run()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Trades produced: {len(trades):,}\n")

    if not trades:
        print("  No trades produced. Phase 4 ends here. Reasons to investigate:")
        print("  - Check Config.DELTA_METHOD = 'ohlc_decomposition'")
        print("  - Run with verbose=True on a known week to see state transitions")
        return

    # Convert to DataFrame and save
    df_trades = trades_to_dataframe(trades)
    csv_path = os.path.join(OUT_DIR, "trades.csv")
    df_trades.to_csv(csv_path, index=False)
    print(f"Trades saved: {csv_path}")

    # ── Headline metrics ──────────────────────────────────────────────────────
    rs        = df_trades['r_multiple'].values
    pnl_pts   = df_trades['pnl_points'].values
    pnl_dol    = df_trades['pnl_dollars'].values
    longs     = df_trades[df_trades['direction'] == 'long']
    shorts    = df_trades[df_trades['direction'] == 'short']
    wins      = df_trades[df_trades['pnl_points'] > 0]
    losses    = df_trades[df_trades['pnl_points'] <= 0]
    by_reason = df_trades.groupby('exit_reason').size().to_dict()

    avg_r        = float(np.mean(rs))
    median_r     = float(np.median(rs))
    sum_r        = float(np.sum(rs))
    win_rate     = len(wins) / len(df_trades) * 100
    pf           = (wins['pnl_points'].sum() / abs(losses['pnl_points'].sum())
                    if len(losses) > 0 and losses['pnl_points'].sum() != 0
                    else float('inf'))
    total_dol     = float(np.sum(pnl_dol))
    total_r_long = float(np.sum(longs['r_multiple']))   if len(longs)  > 0 else 0.0
    total_r_short= float(np.sum(shorts['r_multiple']))  if len(shorts) > 0 else 0.0
    avg_r_long   = float(np.mean(longs['r_multiple']))  if len(longs)  > 0 else 0.0
    avg_r_short  = float(np.mean(shorts['r_multiple'])) if len(shorts) > 0 else 0.0

    # Build summary text
    summary_lines = [
        "="*70,
        "PHASE 4 — CONFIRMATION BACKTEST RESULTS",
        f"{CONFIRMATION_START} to {CONFIRMATION_END}",
        "="*70,
        "",
        "PRE-REGISTERED PARAMETERS (locked 2026-05-05, unchanged):",
        f"  SL_ATR_MULT_TPO     = {Config.SL_ATR_MULT_TPO}",
        f"  MIN_PROBE_DIST_ATR  = {Config.MIN_PROBE_DIST_ATR}",
        f"  VELOCITY_THRESHOLD  = {Config.VELOCITY_THRESHOLD}",
        f"  VA_THRESHOLD        = {Config.VA_THRESHOLD}",
        "",
        "─"*70,
        "TRADE COUNTS",
        "─"*70,
        f"  Total trades              : {len(df_trades):,}",
        f"  Long fades                : {len(longs):,}  ({len(longs)/len(df_trades)*100:.1f}%)",
        f"  Short fades               : {len(shorts):,}  ({len(shorts)/len(df_trades)*100:.1f}%)",
        f"  Wins                      : {len(wins):,}  ({win_rate:.1f}%)",
        f"  Losses                    : {len(losses):,}",
        "",
        "  Exit reasons:",
    ]
    for reason in ['TP', 'SL', 'EOD', 'POC_cancel']:
        n = by_reason.get(reason, 0)
        summary_lines.append(f"    {reason:<14} : {n:,}  ({n/len(df_trades)*100:.1f}%)")

    summary_lines += [
        "",
        "─"*70,
        "R-MULTIPLE METRICS  (1R = SL distance from entry)",
        "─"*70,
        f"  Avg R per trade           : {avg_r:+.4f}    <-- primary test statistic",
        f"  Median R per trade        : {median_r:+.4f}",
        f"  Sum of R                  : {sum_r:+.2f}",
        f"  Std of R                  : {float(np.std(rs)):.4f}",
        "",
        f"  Long  avg R               : {avg_r_long:+.4f}   (n={len(longs):,})",
        f"  Short avg R               : {avg_r_short:+.4f}   (n={len(shorts):,})",
        f"  Long  total R             : {total_r_long:+.2f}",
        f"  Short total R             : {total_r_short:+.2f}",
        "",
        "─"*70,
        "DOLLAR METRICS  (1 ES contract @ $50/point)",
        "─"*70,
        f"  Total P&L                : ${total_dol:>14,.2f}",
        f"  Total points              : {float(np.sum(pnl_pts)):+,.2f}",
        f"  Profit factor             : {pf:.3f}",
        f"  Avg trade $              : ${float(np.mean(pnl_dol)):>14,.2f}",
        "",
    ]

    summary = "\n".join(summary_lines)
    print(summary)

    # Save summary
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(summary)
    print(f"\nSummary saved: {summary_path}")

    # ── Equity curve ──────────────────────────────────────────────────────────
    df_trades = df_trades.sort_values('exit_dt').reset_index(drop=True)
    df_trades['cum_R'] = df_trades['r_multiple'].cumsum()
    df_trades['cum_dol'] = df_trades['pnl_dollars'].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='#131722', sharex=True)
    for ax in axes:
        ax.set_facecolor('#1e222d')
        for sp in ax.spines.values():
            sp.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)
        ax.title.set_color('#d1d4dc')
        ax.xaxis.label.set_color('#d1d4dc')
        ax.yaxis.label.set_color('#d1d4dc')

    axes[0].plot(df_trades['exit_dt'], df_trades['cum_R'],
                 color='#5dade2', linewidth=1.4)
    axes[0].axhline(0, color='#d1d4dc', linewidth=0.6, alpha=0.4)
    axes[0].set_title(f'Cumulative R-Multiple — {len(df_trades):,} trades  '
                      f'final={sum_r:+.2f}R  avg={avg_r:+.4f}R/trade')
    axes[0].set_ylabel('Cumulative R')

    axes[1].plot(df_trades['exit_dt'], df_trades['cum_dol'],
                 color='#26a69a', linewidth=1.4)
    axes[1].axhline(0, color='#d1d4dc', linewidth=0.6, alpha=0.4)
    axes[1].set_title(f'Cumulative $ — 1 contract  final=${total_dol:+,.0f}')
    axes[1].set_ylabel('Cumulative $')
    axes[1].set_xlabel('Exit date')

    fig.suptitle('Phase 4 — Confirmation Backtest', color='#d1d4dc', fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "equity_curve.png")
    fig.savefig(fig_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    plt.show()
    print(f"Equity curve saved: {fig_path}")

    print(f"\n{'='*70}")
    print("Phase 4 complete. Whatever this report shows is THE result.")
    print("If sample is large enough and avg-R looks meaningful (positive or")
    print("negative), proceed to Phase 5: bootstrap p-value, regime-conditional")
    print("sub-tests, and maximum-drawdown distribution.")
    print(f"{'='*70}\n")

    # Restore original delta method
    Config.DELTA_METHOD = saved_delta_method


if __name__ == "__main__":
    main()
