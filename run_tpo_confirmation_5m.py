# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 (5M) — CONFIRMATION BACKTEST AT 5M RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the TPO state machine ONCE on the full 2015–2026 ES dataset using
# the 5M re-pre-registered locked parameters. This is the analogue of
# run_tpo_confirmation.py but at 5M resolution with the new locks.
#
# RE-PRE-REGISTERED PARAMETERS (locked 2026-05-06):
#     SL_ATR_MULT_5M         = 4.544
#     MIN_PROBE_DIST_ATR_5M  = 1.045
#     VELOCITY_THRESHOLD_5M  = 3.808
#
# These were locked AFTER the methodological diagnostic chain established
# that 1H bar resolution was an artifact, but BEFORE running the strategy
# under these new parameters. They do not change after this point regardless
# of the result below.
#
# Outputs (saved to results\confirmation_backtest_5m\):
#     trades.csv           — one row per closed trade with full setup metrics
#     summary.txt          — Table-1-style summary
#     equity_curve.png     — cumulative R-multiple and dollar P&L over time
#
# Runtime expectation: 5–10 minutes (state machine iterates 798k 5M bars).
# ═══════════════════════════════════════════════════════════════════════════════

import os
import time
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

from strategy import Config, compute_delta_ohlc_decomposition
from tpo_state_machine import TPOStateMachine, trades_to_dataframe


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIRMATION_START = "2015-01-01"
CONFIRMATION_END   = "2026-12-31"
TIMEFRAME_FREQ     = '5min'
BARS_PER_DAY       = 276

OUT_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\confirmation_backtest_5m"


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
    print(f"Loading ES 1M data...")
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

    print(f"  Computing OHLC-decomposition delta on 1M...")
    up, down = compute_delta_ohlc_decomposition(df)
    df['delta_1m'] = up - down

    # Filter range
    start = pd.Timestamp(CONFIRMATION_START, tz=CT)
    end   = pd.Timestamp(CONFIRMATION_END,   tz=CT)
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  PHASE 4 (5M) — CONFIRMATION BACKTEST")
    print(f"  {CONFIRMATION_START} → {CONFIRMATION_END}")
    print(f"{'='*72}")
    print(f"\n  RE-PRE-REGISTERED PARAMETERS (locked 2026-05-06):")
    print(f"    SL_ATR_MULT_5M         = {Config.SL_ATR_MULT_5M}")
    print(f"    MIN_PROBE_DIST_ATR_5M  = {Config.MIN_PROBE_DIST_ATR_5M}")
    print(f"    VELOCITY_THRESHOLD_5M  = {Config.VELOCITY_THRESHOLD_5M}")
    print(f"    VA_THRESHOLD           = {Config.VA_THRESHOLD}")
    print(f"\n  These do NOT change based on results below.\n")

    # Load 1M, aggregate to 5M
    df_1m = load_es_1m_data()
    df_5m = aggregate_to_5m(df_1m)

    print(f"\n  Dataset : {len(df_5m):,} 5M bars, {len(df_1m):,} 1M bars\n")

    # Run state machine
    print("  Running TPO state machine over full 5M range...")
    t0 = time.time()
    sm = TPOStateMachine(
        df_1h        = df_5m,             # 5M bars feed the state-machine loop
        df_1m        = df_1m,             # 1M bars feed the engine's profile
        contracts    = 1,
        verbose      = False,
        gate_enabled = True,              # pre-registered ON
        entry_mode   = 'retest',          # pre-registered retest entry
        bar_freq     = TIMEFRAME_FREQ,
        bars_per_day = BARS_PER_DAY,
        param_set    = '5m',              # ← KEY: use the 5M locks
    )
    trades = sm.run()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Trades: {len(trades):,}\n")

    if not trades:
        print("  No trades produced. Phase 4 (5M) ends here.")
        return

    df_trades = trades_to_dataframe(trades)
    csv_path = os.path.join(OUT_DIR, "trades.csv")
    df_trades.to_csv(csv_path, index=False)
    print(f"  Trades saved: {csv_path}")

    # ── Headline metrics ──────────────────────────────────────────────────────
    rs        = df_trades['r_multiple'].values
    pnl_pts   = df_trades['pnl_points'].values
    pnl_dol   = df_trades['pnl_dollars'].values
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
    total_dol    = float(np.sum(pnl_dol))
    avg_r_long   = float(np.mean(longs['r_multiple']))  if len(longs)  > 0 else 0.0
    avg_r_short  = float(np.mean(shorts['r_multiple'])) if len(shorts) > 0 else 0.0
    total_r_long = float(np.sum(longs['r_multiple']))   if len(longs)  > 0 else 0.0
    total_r_short= float(np.sum(shorts['r_multiple']))  if len(shorts) > 0 else 0.0

    summary_lines = [
        "="*72,
        "PHASE 4 (5M) — CONFIRMATION BACKTEST RESULTS",
        f"{CONFIRMATION_START} to {CONFIRMATION_END}",
        "="*72,
        "",
        "RE-PRE-REGISTERED PARAMETERS (locked 2026-05-06, unchanged):",
        f"  SL_ATR_MULT_5M         = {Config.SL_ATR_MULT_5M}",
        f"  MIN_PROBE_DIST_ATR_5M  = {Config.MIN_PROBE_DIST_ATR_5M}",
        f"  VELOCITY_THRESHOLD_5M  = {Config.VELOCITY_THRESHOLD_5M}",
        f"  VA_THRESHOLD           = {Config.VA_THRESHOLD}",
        "",
        "─"*72,
        "TRADE COUNTS",
        "─"*72,
        f"  Total trades              : {len(df_trades):,}",
        f"  Long fades                : {len(longs):,}  "
        f"({len(longs)/len(df_trades)*100:.1f}%)",
        f"  Short fades               : {len(shorts):,}  "
        f"({len(shorts)/len(df_trades)*100:.1f}%)",
        f"  Wins                      : {len(wins):,}  ({win_rate:.1f}%)",
        f"  Losses                    : {len(losses):,}",
        "",
        "  Exit reasons:",
    ]
    for reason in ['TP', 'SL', 'EOD', 'POC_cancel']:
        n = by_reason.get(reason, 0)
        summary_lines.append(f"    {reason:<14} : {n:,}  "
                             f"({n/len(df_trades)*100:.1f}%)")

    summary_lines += [
        "",
        "─"*72,
        "R-MULTIPLE METRICS  (1R = SL distance from entry)",
        "─"*72,
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
        "─"*72,
        "DOLLAR METRICS  (1 ES contract @ $50/point)",
        "─"*72,
        f"  Total P&L                 : ${total_dol:>14,.2f}",
        f"  Total points              : {float(np.sum(pnl_pts)):+,.2f}",
        f"  Profit factor             : {pf:.3f}",
        f"  Avg trade $               : ${float(np.mean(pnl_dol)):>14,.2f}",
        "",
    ]

    summary = "\n".join(summary_lines)
    print(summary)

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(summary)
    print(f"\nSummary saved: {summary_path}")

    # ── Equity curve ──────────────────────────────────────────────────────────
    df_trades = df_trades.sort_values('exit_dt').reset_index(drop=True)
    df_trades['cum_R']   = df_trades['r_multiple'].cumsum()
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

    fig.suptitle('Phase 4 (5M) — Confirmation Backtest', color='#d1d4dc', fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "equity_curve.png")
    fig.savefig(fig_path, dpi=120, facecolor='#131722', bbox_inches='tight')
    plt.show()
    print(f"Equity curve saved: {fig_path}")

    print(f"\n{'='*72}")
    print("  Phase 4 (5M) complete. Whatever this report shows is THE result.")
    print("  Next: Phase 5b (5M) cross-instrument replication on NQ/GC/ZN.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
