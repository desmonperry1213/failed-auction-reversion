# ═══════════════════════════════════════════════════════════════════════════════
# TPO ENGINE — Volume-at-Price Profile for Weekly Auction Fade Hypothesis
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase 1 deliverable per IB_Auction_Fade_Framework.pdf and TRANSITION_PROMPT_V2.
#
# Implements:
#   - TPOEngine class: developing weekly + daily Value Areas from 1M OHLCV
#   - view_tpo_week(): visual validation tool (heatmap, candles, overlap state)
#
# This is a STANDALONE module so existing strategy.py runs unchanged. Once
# the engine is visually validated on at least 5 known weeks (per Phase 1
# acceptance criterion in the framework), merge the TPOEngine class into
# strategy.py as Section 3.5 (before InitialBalance) and view_tpo_week into
# Section 7. Until then, keep this separate to avoid touching 1846 lines of
# working IB code.
#
# Usage (see test_tpo.py for full runner):
#     from strategy import load_and_prepare_data, Config
#     from tpo_engine import view_tpo_week
#     df_1h, df_1m = load_and_prepare_data()
#     view_tpo_week(df_1h, df_1m, "2026-04-26", "2026-05-03")
#
# DESIGN DECISIONS LOCKED PER FRAMEWORK:
#   - VA threshold: 70% of weekly accumulated volume (1 std dev, theory-derived)
#   - Tick resolution: 0.25 (ES tick size; pulled from Config.TICK_SIZE)
#   - Volume distribution: uniform across each 1M bar's tick range (low → high)
#   - Weekly reset: first bar with weekday()==6 following a non-Sunday bar
#                   (Sunday Globex open ≈ 17:00 CT)
#   - Daily reset:  bar at 17:00 CT (start of new Globex session day)
#                   Section 7 of the transition prompt mentioned 00:00 CT but
#                   Section 10 specified 17:00 CT explicitly with reasoning.
#                   17:00 CT chosen here. Override via Config.DAILY_RESET_HOUR.
#   - Holiday handling: engine accumulates on every bar; holiday-suspension
#                       gates execution only and lives in the state machine.
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import pytz

from strategy import Config


# Inject a default for the daily reset hour if the host Config does not yet
# define one. This keeps tpo_engine.py drop-in compatible with the current
# strategy.py while allowing override later.
if not hasattr(Config, 'DAILY_RESET_HOUR'):
    Config.DAILY_RESET_HOUR = 17  # 17:00 CT = Globex session day boundary
if not hasattr(Config, 'VA_THRESHOLD'):
    Config.VA_THRESHOLD = 0.70    # 70% of accumulated volume (1 std dev)
if not hasattr(Config, 'TPO_ENABLED'):
    Config.TPO_ENABLED = False    # Mode switch; engine is opt-in for state machine


# ═══════════════════════════════════════════════════════════════════════════════
# TPO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TPOEngine:
    """
    Maintains two volume-at-price profiles, updated incrementally as 1H bars
    arrive together with their constituent 1M bars:

        weekly_profile : {tick_index: volume}  — reset at Sunday Globex open
        daily_profile  : {tick_index: volume}  — reset at each Globex session day

    After every update, the developing 70% Value Area is recomputed for both
    profiles. Levels are exposed as properties:

        weekly_vah, weekly_val, weekly_poc
        daily_vah,  daily_val,  daily_poc

    Tick indices are stored as integers (price / tick_size, rounded) to avoid
    floating-point key collisions. Conversion back to price uses tick_size.

    Volume-at-price construction:
        For each 1M bar, distribute bar.volume uniformly across all ticks
        between bar.low and bar.high (inclusive) at TICK_SIZE resolution.
        Each tick receives bar.volume / num_ticks_in_range.

    Reset triggers:
        Weekly: bar.weekday() == 6 (Sunday) AND prev bar weekday != 6
                → catches the Sunday 17:00 CT bar that opens the new week
        Daily:  bar.hour == DAILY_RESET_HOUR AND bar.minute == 0
                → 17:00 CT by default (start of new Globex session day)
    """

    def __init__(self, tick_size=None, va_threshold=None, daily_reset_hour=None):
        self.tick_size        = tick_size        if tick_size        is not None else Config.TICK_SIZE
        self.va_threshold     = va_threshold     if va_threshold     is not None else Config.VA_THRESHOLD
        self.daily_reset_hour = daily_reset_hour if daily_reset_hour is not None else Config.DAILY_RESET_HOUR

        # Profiles: {tick_int: cumulative_volume}
        self.weekly_profile = {}
        self.daily_profile  = {}

        # Reset bookkeeping
        self.weekly_reset_dt = None
        self.daily_reset_dt  = None
        self._prev_bar_dt    = None

        # Cached level values (in price units)
        self._weekly_vah = None
        self._weekly_val = None
        self._weekly_poc = None
        self._daily_vah  = None
        self._daily_val  = None
        self._daily_poc  = None

        # Snapshots for visualization. One row per call to add_bar().
        # Each entry: dict with datetime + level fields + overlap flag.
        self.snapshots = []

    # ── Reset methods ─────────────────────────────────────────────────────────

    def reset_weekly(self, dt_now):
        self.weekly_profile  = {}
        self.weekly_reset_dt = dt_now
        self._weekly_vah     = None
        self._weekly_val     = None
        self._weekly_poc     = None

    def reset_daily(self, dt_now):
        self.daily_profile  = {}
        self.daily_reset_dt = dt_now
        self._daily_vah     = None
        self._daily_val     = None
        self._daily_poc     = None

    # ── Reset detection ───────────────────────────────────────────────────────

    def _check_resets(self, bar_dt):
        # ── Weekly reset: first Sunday bar after a non-Sunday bar ─────────────
        if self._prev_bar_dt is None:
            # First bar ever in this engine's life
            if bar_dt.weekday() == 6:
                self.reset_weekly(bar_dt)
        else:
            if bar_dt.weekday() == 6 and self._prev_bar_dt.weekday() != 6:
                self.reset_weekly(bar_dt)

        # ── Daily reset: bar at DAILY_RESET_HOUR (17:00 CT default) ──────────
        if bar_dt.hour == self.daily_reset_hour and bar_dt.minute == 0:
            self.reset_daily(bar_dt)
        elif self.daily_reset_dt is None:
            # Initialize on very first bar so daily_reset_dt is never None
            # while daily_profile holds data.
            self.daily_reset_dt = bar_dt

    # ── Bar update ────────────────────────────────────────────────────────────

    def add_bar(self, bar_1h, minutes_df):
        """
        Update both profiles with one 1H bar's constituent 1M bars.

        bar_1h     : pandas Series with 'datetime', 'open', 'high', 'low', 'close'
        minutes_df : DataFrame of 1M bars within this hour (or None / empty)
        """
        bar_dt = bar_1h['datetime']
        self._check_resets(bar_dt)

        # Distribute volume of each 1M bar uniformly across its tick range
        if minutes_df is not None and len(minutes_df) > 0:
            for _, m in minutes_df.iterrows():
                self._add_minute_bar(m)

        # Recompute developing VAs after this hour's worth of volume
        self._weekly_vah, self._weekly_val, self._weekly_poc = self._compute_va(self.weekly_profile)
        self._daily_vah,  self._daily_val,  self._daily_poc  = self._compute_va(self.daily_profile)

        # Snapshot
        self.snapshots.append({
            'datetime'  : bar_dt,
            'weekly_vah': self._weekly_vah,
            'weekly_val': self._weekly_val,
            'weekly_poc': self._weekly_poc,
            'daily_vah' : self._daily_vah,
            'daily_val' : self._daily_val,
            'daily_poc' : self._daily_poc,
            'overlap'   : self.daily_poc_in_weekly_va(),
        })

        self._prev_bar_dt = bar_dt

    def _add_minute_bar(self, m):
        vol = float(m.get('volume', 0))
        if vol <= 0 or not np.isfinite(vol):
            return
        low  = float(m['low'])
        high = float(m['high'])
        if not (np.isfinite(low) and np.isfinite(high)):
            return

        low_tick  = int(round(low  / self.tick_size))
        high_tick = int(round(high / self.tick_size))
        if high_tick < low_tick:
            low_tick, high_tick = high_tick, low_tick

        n_ticks      = high_tick - low_tick + 1
        vol_per_tick = vol / n_ticks

        for t in range(low_tick, high_tick + 1):
            self.weekly_profile[t] = self.weekly_profile.get(t, 0.0) + vol_per_tick
            self.daily_profile[t]  = self.daily_profile.get(t,  0.0) + vol_per_tick

    # ── VA algorithm (standard 70% expansion from POC outward) ────────────────

    def _compute_va(self, profile):
        """
        Returns (vah, val, poc) in price units, or (None, None, None) if empty.

        Algorithm:
            1. POC = tick with highest volume (ties broken by lowest tick)
            2. Initialize range = [POC, POC], captured = profile[POC]
            3. While captured < 70% of total volume:
                   compare volume of next tick above vs next tick below
                   add the larger one (tie → up) and accumulate its volume
            4. VAH = top tick × tick_size, VAL = bottom tick × tick_size
        """
        if not profile:
            return (None, None, None)
        total_vol = sum(profile.values())
        if total_vol <= 0:
            return (None, None, None)

        # POC selection: highest volume; deterministic tie-break by lowest tick
        poc_tick  = max(profile, key=lambda t: (profile[t], -t))
        poc_price = poc_tick * self.tick_size

        target_vol = self.va_threshold * total_vol
        captured   = profile[poc_tick]
        low_tick   = poc_tick
        high_tick  = poc_tick

        all_ticks = profile.keys()
        min_tick  = min(all_ticks)
        max_tick  = max(all_ticks)

        # Edge case: profile only spans one tick
        if min_tick == max_tick:
            return (poc_price, poc_price, poc_price)

        # Expand outward
        while captured < target_vol:
            above_idx = high_tick + 1
            below_idx = low_tick  - 1

            can_go_up   = above_idx <= max_tick
            can_go_down = below_idx >= min_tick
            if not can_go_up and not can_go_down:
                break

            vol_up   = profile.get(above_idx, 0.0) if can_go_up   else float('-inf')
            vol_down = profile.get(below_idx, 0.0) if can_go_down else float('-inf')

            if vol_up >= vol_down:
                high_tick = above_idx
                captured += max(vol_up, 0.0)
            else:
                low_tick  = below_idx
                captured += max(vol_down, 0.0)

        return (high_tick * self.tick_size,
                low_tick  * self.tick_size,
                poc_price)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def weekly_vah(self): return self._weekly_vah
    @property
    def weekly_val(self): return self._weekly_val
    @property
    def weekly_poc(self): return self._weekly_poc
    @property
    def daily_vah(self):  return self._daily_vah
    @property
    def daily_val(self):  return self._daily_val
    @property
    def daily_poc(self):  return self._daily_poc

    @property
    def weekly_total_range(self):
        if self._weekly_vah is None or self._weekly_val is None:
            return None
        return self._weekly_vah - self._weekly_val

    def daily_poc_in_weekly_va(self):
        """
        Execution gate per the framework: daily developing POC must fall within
        the weekly developing VA at the time of trade entry. Returns False if
        either profile is uninitialized.
        """
        if (self._daily_poc is None or
            self._weekly_vah is None or self._weekly_val is None):
            return False
        return self._weekly_val <= self._daily_poc <= self._weekly_vah

    # ── Heatmap export (for visualization) ────────────────────────────────────

    def get_weekly_profile_array(self):
        """
        Returns (prices, volumes) parallel arrays of the current weekly profile,
        sorted ascending by price. For horizontal-histogram overlay.
        """
        if not self.weekly_profile:
            return np.array([]), np.array([])
        ticks   = sorted(self.weekly_profile.keys())
        prices  = np.array([t * self.tick_size for t in ticks])
        volumes = np.array([self.weekly_profile[t] for t in ticks])
        return prices, volumes

    def get_daily_profile_array(self):
        """Same as get_weekly_profile_array() but for the daily profile."""
        if not self.daily_profile:
            return np.array([]), np.array([])
        ticks   = sorted(self.daily_profile.keys())
        prices  = np.array([t * self.tick_size for t in ticks])
        volumes = np.array([self.daily_profile[t] for t in ticks])
        return prices, volumes


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW_TPO_WEEK — Visual validation tool
# ═══════════════════════════════════════════════════════════════════════════════

def view_tpo_week(df_1h, df_1m, week_start, week_end, save_path=None, show=True):
    """
    Visual validation of the TPO engine over a single week (or any window).

    Produces a 4-panel matplotlib figure on a TradingView-style dark theme:

        Top-left:    1H candlesticks with developing weekly VAH/VAL/POC and
                     daily VAH/VAL/POC overlaid as time-evolving lines.
        Top-right:   Final weekly volume-at-price histogram (horizontal bars)
                     with VAH/VAL/POC markers. Y-axis aligned with the candles.
        Bottom-left: Step plot of daily-POC-in-weekly-VA (the execution gate).
        Bottom-right: Summary table.

    Validation checklist (use this on at least 5 known weeks before Phase 2):
        1. POC visually corresponds to the densest price level on the histogram
        2. VAH / VAL bracket roughly the central 70% of the histogram mass
        3. Developing levels stabilize as the week progresses (more data → less drift)
        4. Daily POC moves more than weekly POC (shorter window → more variance)
        5. Overlap step plot toggles consistently with daily POC crossing weekly VA

    Inputs:
        df_1h       : full 1H DataFrame from load_and_prepare_data()
        df_1m       : full 1M DataFrame from load_and_prepare_data()
        week_start  : "YYYY-MM-DD" inclusive
        week_end    : "YYYY-MM-DD" inclusive
        save_path   : optional PNG output path
        show        : if True, show the figure interactively

    Returns the TPOEngine instance for further inspection.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    CT = pytz.timezone(Config.TIMEZONE)
    s  = pd.Timestamp(week_start, tz=CT)
    # End of day inclusive: bump to start of next day
    e  = pd.Timestamp(week_end, tz=CT) + pd.Timedelta(days=1)

    win_1h = df_1h[(df_1h['datetime'] >= s) & (df_1h['datetime'] < e)].copy().reset_index(drop=True)
    win_1m = df_1m[(df_1m['datetime'] >= s) & (df_1m['datetime'] < e)].copy().reset_index(drop=True)
    win_1m['hour_bucket'] = win_1m['datetime'].dt.floor('h')

    if len(win_1h) == 0:
        print(f"[view_tpo_week] No 1H bars in window {week_start} → {week_end}")
        return None

    # ── Run engine across the window ──────────────────────────────────────────
    engine = TPOEngine()
    for _, bar in win_1h.iterrows():
        bar_minutes = win_1m[win_1m['hour_bucket'] == bar['datetime']]
        engine.add_bar(bar, bar_minutes)
    snaps = pd.DataFrame(engine.snapshots)

    # ── Figure scaffold ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9), facecolor='#131722')
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[5, 1.2],
        height_ratios=[4, 1],
        hspace=0.08, wspace=0.03,
    )
    ax_price   = fig.add_subplot(gs[0, 0], facecolor='#1e222d')
    ax_profile = fig.add_subplot(gs[0, 1], facecolor='#1e222d', sharey=ax_price)
    ax_overlap = fig.add_subplot(gs[1, 0], facecolor='#1e222d', sharex=ax_price)
    ax_summary = fig.add_subplot(gs[1, 1], facecolor='#1e222d')

    for ax in (ax_price, ax_profile, ax_overlap, ax_summary):
        for spine in ax.spines.values():
            spine.set_color('#2a2e39')
        ax.tick_params(colors='#d1d4dc', labelsize=8)
        ax.grid(True, color='#2a2e39', linewidth=0.4, alpha=0.6)

    # ── Top-left: 1H candles + developing levels ──────────────────────────────
    times_num = mdates.date2num(win_1h['datetime'])
    bar_w     = (times_num[-1] - times_num[0]) / max(len(win_1h) * 1.6, 1) if len(win_1h) > 1 else 0.02
    for i, row in win_1h.iterrows():
        x     = times_num[i]
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        # Wick
        ax_price.plot([x, x], [row['low'], row['high']], color=color, linewidth=0.8, zorder=2)
        # Body
        body_lo = min(row['open'], row['close'])
        body_h  = max(abs(row['close'] - row['open']), 0.05)
        ax_price.add_patch(Rectangle(
            (x - bar_w/2, body_lo), bar_w, body_h,
            facecolor=color, edgecolor=color, zorder=3,
        ))

    # Developing weekly levels (stronger styling)
    snap_times = mdates.date2num(snaps['datetime'])
    ax_price.plot(snap_times, snaps['weekly_vah'], color='#5dade2', linewidth=1.4,
                  linestyle='--', label='Weekly VAH', zorder=4)
    ax_price.plot(snap_times, snaps['weekly_val'], color='#5dade2', linewidth=1.4,
                  linestyle='--', label='Weekly VAL', zorder=4)
    ax_price.plot(snap_times, snaps['weekly_poc'], color='#f1c40f', linewidth=1.8,
                  label='Weekly POC', zorder=5)
    # Developing daily levels (lighter)
    ax_price.plot(snap_times, snaps['daily_vah'], color='#bb6bd9', linewidth=0.9,
                  alpha=0.7, label='Daily VAH', zorder=4)
    ax_price.plot(snap_times, snaps['daily_val'], color='#bb6bd9', linewidth=0.9,
                  alpha=0.7, label='Daily VAL', zorder=4)
    ax_price.plot(snap_times, snaps['daily_poc'], color='#e74c3c', linewidth=1.0,
                  alpha=0.8, label='Daily POC', zorder=5)

    leg = ax_price.legend(
        loc='upper left', facecolor='#1e222d', edgecolor='#2a2e39',
        labelcolor='#d1d4dc', fontsize=8, ncol=2,
    )
    ax_price.set_title(
        f"TPO Engine Validation — {week_start} to {week_end}",
        color='#d1d4dc', fontsize=12, pad=10,
    )
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d %H:%M'))
    plt.setp(ax_price.get_xticklabels(), visible=False)
    ax_price.set_ylabel('Price', color='#d1d4dc', fontsize=9)

    # ── Top-right: final weekly volume profile ────────────────────────────────
    prices, volumes = engine.get_weekly_profile_array()
    if len(prices) > 0:
        ax_profile.barh(
            prices, volumes,
            height=Config.TICK_SIZE,
            color='#5dade2', alpha=0.55, edgecolor='none',
        )
        if engine.weekly_vah is not None:
            ax_profile.axhline(engine.weekly_vah, color='#5dade2', linestyle='--', linewidth=1.2)
            ax_profile.axhline(engine.weekly_val, color='#5dade2', linestyle='--', linewidth=1.2)
            ax_profile.axhline(engine.weekly_poc, color='#f1c40f', linestyle='-',  linewidth=1.5)
    total_wkly_vol = int(sum(engine.weekly_profile.values())) if engine.weekly_profile else 0
    ax_profile.set_title(
        f"Weekly Profile\nTotal vol: {total_wkly_vol:,}",
        color='#d1d4dc', fontsize=9, pad=8,
    )
    plt.setp(ax_profile.get_yticklabels(), visible=False)
    ax_profile.set_xlabel('Volume', color='#d1d4dc', fontsize=8)

    # ── Bottom-left: overlap state ────────────────────────────────────────────
    overlap_int = snaps['overlap'].astype(int).values
    ax_overlap.fill_between(
        snap_times, 0, overlap_int,
        step='post', color='#26a69a', alpha=0.55,
        label='Daily POC inside Weekly VA',
    )
    ax_overlap.set_ylim(-0.1, 1.15)
    ax_overlap.set_yticks([0, 1])
    ax_overlap.set_yticklabels(['No', 'Yes'])
    ax_overlap.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d %H:%M'))
    plt.setp(ax_overlap.get_xticklabels(), rotation=30, ha='right', fontsize=7)
    ax_overlap.legend(
        loc='upper left', facecolor='#1e222d', edgecolor='#2a2e39',
        labelcolor='#d1d4dc', fontsize=8,
    )
    ax_overlap.set_ylabel('Gate', color='#d1d4dc', fontsize=9)

    # ── Bottom-right: summary text ────────────────────────────────────────────
    ax_summary.axis('off')
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) and v is not None else "—"
    overlap_pct = 100.0 * overlap_int.mean() if len(overlap_int) > 0 else 0.0
    summary_lines = [
        f"Bars 1H : {len(win_1h)}    1M : {len(win_1m):,}",
        "",
        "WEEKLY (final)",
        f"  VAH : {fmt(engine.weekly_vah)}",
        f"  POC : {fmt(engine.weekly_poc)}",
        f"  VAL : {fmt(engine.weekly_val)}",
        f"  Rng : {fmt(engine.weekly_total_range)}",
        "",
        "DAILY (final)",
        f"  VAH : {fmt(engine.daily_vah)}",
        f"  POC : {fmt(engine.daily_poc)}",
        f"  VAL : {fmt(engine.daily_val)}",
        "",
        f"Overlap %: {overlap_pct:.1f}%",
    ]
    ax_summary.text(
        0.02, 0.97, "\n".join(summary_lines),
        transform=ax_summary.transAxes,
        color='#d1d4dc', fontsize=9, fontfamily='monospace',
        verticalalignment='top',
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, facecolor='#131722', bbox_inches='tight')
        print(f"[view_tpo_week] Saved figure: {save_path}")

    if show:
        plt.show()

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  TPO VALIDATION — {week_start} to {week_end}")
    print(f"{'─'*64}")
    print(f"  1H bars in window     : {len(win_1h)}")
    print(f"  1M bars in window     : {len(win_1m):,}")
    print(f"  Weekly profile ticks  : {len(engine.weekly_profile)}")
    print(f"  Daily  profile ticks  : {len(engine.daily_profile)}")
    print(f"  Final weekly VAH      : {fmt(engine.weekly_vah)}")
    print(f"  Final weekly POC      : {fmt(engine.weekly_poc)}")
    print(f"  Final weekly VAL      : {fmt(engine.weekly_val)}")
    print(f"  Final weekly range    : {fmt(engine.weekly_total_range)}")
    print(f"  Final daily  VAH      : {fmt(engine.daily_vah)}")
    print(f"  Final daily  POC      : {fmt(engine.daily_poc)}")
    print(f"  Final daily  VAL      : {fmt(engine.daily_val)}")
    print(f"  Daily POC in weekly VA: {engine.daily_poc_in_weekly_va()}")
    print(f"  Overlap pct over week : {overlap_pct:.1f}%")
    print(f"  Total weekly vol      : {total_wkly_vol:,}")
    print(f"{'─'*64}\n")
    return engine
