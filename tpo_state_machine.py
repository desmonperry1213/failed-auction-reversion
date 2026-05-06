# ═══════════════════════════════════════════════════════════════════════════════
# TPO STATE MACHINE — Failed Weekly Auction Detection
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase 3 deliverable per IB_Auction_Fade_Framework.pdf and TRANSITION_PROMPT_V2.
#
# Implements the 6-state sequence specified in framework Section 3.4 using the
# weekly VA/POC produced by tpo_engine.TPOEngine. Replaces the IB-based logic
# in strategy.py without touching it — old strategy still runs unchanged when
# Config.TPO_ENABLED = False, new strategy runs when True.
#
# LOCKED PRE-REGISTERED PARAMETERS  (recorded 2026-05-05):
#     SL_ATR_MULT_TPO       = 1.578   median of distance/ATR distribution
#     MIN_PROBE_DIST_ATR    = 0.40    10th percentile (no natural break in dist)
#     VELOCITY_THRESHOLD    = 2.21    90th percentile (start of spike-zone tail)
#
# These DO NOT change after this point. Source: pre_registration_analysis.py
# run on full 2015–2026 dataset, 30,487 probes. Stability evidence:
# distance/ATR median spread across sub-periods = 0.036 (stable). Velocity
# spread of 0.232 attributed to global-median artifact; production uses
# rolling-60-day median for the volume normalization, which removes it.
#
# STATE DEFINITIONS:
#     0 IDLE              — before first weekly VA forms (engine warmup)
#     1 WATCHING          — VA exists, price inside, monitoring for close outside
#     2 PROBE_PENDING     — close outside VA, waiting for delta + quality filters
#     3 PROBE_CONFIRMED   — delta divergence + quality passed; waiting for back-inside
#     4 RETEST_WAITING    — back inside; daily VA overlap gate passed; waiting for retest
#     5 IN_TRADE          — limit filled; managing position
#
# CONTINUOUS CHECKS (run every bar in states 2/3/4):
#     - VA acceptance: if weekly VA migrates to engulf the probe extreme,
#                      reset to State 1 (the extension was accepted, no fade)
#     - POC cancellation (state 4 only): if any bar's wick touches weekly POC
#                      before fill, cancel the setup and reset
#
# TRADE MANAGEMENT (per framework — no TP1/TP2 split, single full exit):
#     Entry  : limit at VAH (short) / VAL (long) when bar comes within 4 ticks
#     SL     : probe_extreme ± SL_ATR_MULT_TPO × ATR(14)
#     TP     : weekly POC at the time of entry (full position, single exit)
#     EOD    : flatten at 16:00 CT regardless of P&L
#     No breakeven, no trailing stop, no partial exit
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from strategy import Config, is_full_session
from tpo_engine import TPOEngine


# ── Inject locked pre-registered parameters into Config ───────────────────────
if not hasattr(Config, 'SL_ATR_MULT_TPO'):
    Config.SL_ATR_MULT_TPO       = 1.578
if not hasattr(Config, 'MIN_PROBE_DIST_ATR'):
    Config.MIN_PROBE_DIST_ATR    = 0.40
if not hasattr(Config, 'VELOCITY_THRESHOLD'):
    Config.VELOCITY_THRESHOLD    = 2.21

# 5M re-pre-registration locks (locked 2026-05-06)
if not hasattr(Config, 'SL_ATR_MULT_5M'):
    Config.SL_ATR_MULT_5M        = 4.544
if not hasattr(Config, 'MIN_PROBE_DIST_ATR_5M'):
    Config.MIN_PROBE_DIST_ATR_5M = 1.045
if not hasattr(Config, 'VELOCITY_THRESHOLD_5M'):
    Config.VELOCITY_THRESHOLD_5M = 3.808

# ── Production-only configuration (not part of pre-registration) ──────────────
if not hasattr(Config, 'ROLLING_VOLUME_DAYS'):
    Config.ROLLING_VOLUME_DAYS    = 60   # for rolling median in velocity score
if not hasattr(Config, 'RETEST_PROXIMITY_TICKS'):
    Config.RETEST_PROXIMITY_TICKS = 4    # backtest-fill approximation
if not hasattr(Config, 'EOD_FLATTEN_HOUR'):
    Config.EOD_FLATTEN_HOUR       = 16   # 16:00 CT
if not hasattr(Config, 'POINT_VALUE'):
    Config.POINT_VALUE            = 50.0 # ES = $50/point


# State name lookup for diagnostics
STATE_NAMES = {
    0: 'IDLE',
    1: 'WATCHING',
    2: 'PROBE_PENDING',
    3: 'PROBE_CONFIRMED',
    4: 'RETEST_WAITING',
    5: 'IN_TRADE',
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE RECORD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TPOTrade:
    """One closed trade. Captures both P&L and full setup characteristics
    so Paper 2's classifier has the features it needs without re-running."""
    # Core
    entry_dt        : Any            # datetime
    direction       : str            # 'long' or 'short'
    entry_price     : float
    sl              : float
    tp_poc          : float
    exit_dt         : Any
    exit_price      : float
    exit_reason     : str            # 'TP', 'SL', 'EOD', 'POC_cancel'
    pnl_points      : float
    pnl_dollars     : float          # 1 contract × $50/point
    contracts       : int

    # Setup metrics at entry
    atr_at_entry        : float
    probe_extreme       : float
    probe_distance_atr  : float
    velocity_score      : float
    weekly_vah_at_entry : float
    weekly_val_at_entry : float
    weekly_poc_at_entry : float
    daily_vah_at_entry  : float
    daily_val_at_entry  : float
    daily_poc_at_entry  : float
    hrs_into_week       : float

    # Sequence timestamps for diagnostics
    probe_dt            : Any
    delta_confirm_dt    : Any
    back_inside_dt      : Any

    # R-multiple (computed post-fact)
    r_multiple          : float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — ATR and rolling volume
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df_1h, period=14):
    """Standard 14-period ATR on hourly bars."""
    h = df_1h['high'].values
    l = df_1h['low'].values
    c = df_1h['close'].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


def compute_rolling_volume_median(df_1h, days, bars_per_day=23):
    """Rolling 60-day (default) median of hourly volume.
    23 hours per Globex day for 1H bars — adjust bars_per_day for other timeframes.
    Floor at 100 bars to ensure min_periods works."""
    window_bars = max(int(days * bars_per_day), 100)
    return df_1h['volume'].rolling(window_bars, min_periods=100).median().values


# ═══════════════════════════════════════════════════════════════════════════════
# TPO STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class TPOStateMachine:
    """
    Phase 3 state machine. Walks 1H bars, maintains a TPOEngine in lockstep,
    and produces one TPOTrade per closed position. Single exit at POC, no
    TP1/TP2 split, no breakeven.

    Usage:
        sm = TPOStateMachine(df_1h, df_1m)
        trades = sm.run()
    """

    def __init__(self, df_1h, df_1m, contracts=1, verbose=False,
                 gate_enabled=True, entry_mode='retest',
                 bar_freq='h', bars_per_day=23,
                 param_set='1h'):
        # ─────────────────────────────────────────────────────────────────────
        # NOTE: `gate_enabled`, `entry_mode`, `bar_freq`, `bars_per_day`, and
        # `param_set` are diagnostic/configuration flags. Defaults preserve
        # original 1H pre-registered behavior exactly.
        #
        # `param_set` selects which locked parameter set to use:
        #   '1h' → SL=1.578, min_dist=0.40,  vel=2.21    (original 2026-05-05)
        #   '5m' → SL=4.544, min_dist=1.045, vel=3.808   (re-pre-reg 2026-05-06)
        #
        # Diagnostic alternatives (not pre-registered):
        #   gate_enabled=False         → bypass daily-VA gate
        #   entry_mode='back_inside'   → enter at back-inside bar's close
        #   bar_freq='5min'/'15min'/etc → run on alternative timeframe
        #   bars_per_day=N             → for rolling-volume window scaling
        # ─────────────────────────────────────────────────────────────────────
        self.df_1h    = df_1h.copy().reset_index(drop=True)
        self.df_1m    = df_1m.copy()
        self.contracts = contracts
        self.verbose  = verbose
        self.gate_enabled = gate_enabled
        self.entry_mode   = entry_mode
        self.bar_freq     = bar_freq
        self.bars_per_day = bars_per_day
        if entry_mode not in ('retest', 'back_inside'):
            raise ValueError(f"entry_mode must be 'retest' or 'back_inside', got {entry_mode!r}")

        # Resolve locked parameter set
        if param_set == '1h':
            self.sl_atr_mult       = Config.SL_ATR_MULT_TPO
            self.min_probe_dist_atr = Config.MIN_PROBE_DIST_ATR
            self.velocity_threshold = Config.VELOCITY_THRESHOLD
        elif param_set == '5m':
            self.sl_atr_mult       = Config.SL_ATR_MULT_5M
            self.min_probe_dist_atr = Config.MIN_PROBE_DIST_ATR_5M
            self.velocity_threshold = Config.VELOCITY_THRESHOLD_5M
        else:
            raise ValueError(f"param_set must be '1h' or '5m', got {param_set!r}")
        self.param_set = param_set

        # Diagnostic counters — populated always, used externally
        self.counters = {
            'closes_outside_va'    : 0,
            'closed_back_no_conf'  : 0,
            'delta_divergences'    : 0,
            'failed_min_distance'  : 0,
            'passed_min_distance'  : 0,
            'failed_velocity'      : 0,
            'passed_velocity'      : 0,
            'probe_confirmed'      : 0,
            'back_inside_events'   : 0,
            'gate_passed'          : 0,
            'gate_failed'          : 0,
            'retest_started'       : 0,
            'retest_filled'        : 0,
            'immediate_entries'    : 0,
            'poc_cancellations'    : 0,
            'va_acceptance_resets' : 0,
            'trades_closed'        : 0,
        }

        # Precompute hourly indicators
        self.df_1h['atr']           = compute_atr(self.df_1h, period=Config.ATR_PERIOD)
        self.df_1h['rolling_vol']   = compute_rolling_volume_median(
            self.df_1h, days=Config.ROLLING_VOLUME_DAYS, bars_per_day=self.bars_per_day
        )
        self.df_1h['range']         = self.df_1h['high'] - self.df_1h['low']

        # Index 1M bars by strategy-bar bucket for fast engine updates
        self.df_1m['hour_bucket'] = self.df_1m['datetime'].dt.floor(self.bar_freq)
        self._m_index = {h: g.reset_index(drop=True)
                         for h, g in self.df_1m.groupby('hour_bucket')}

        # Engine
        self.engine = TPOEngine()

        # State
        self.state            = 0  # IDLE
        self.probe_direction  = None       # 'above' or 'below'
        self.probe_extreme    = None       # furthest point reached
        self.probe_dt         = None
        self.delta_confirm_dt = None
        self.back_inside_dt   = None

        # Setup snapshot (frozen at moment of delta confirmation, used at entry)
        self.snapshot_atr        = None
        self.snapshot_velocity   = None
        self.snapshot_probe_dist = None
        self.snapshot_weekly_vah = None
        self.snapshot_weekly_val = None
        self.snapshot_weekly_poc = None
        self.snapshot_daily_vah  = None
        self.snapshot_daily_val  = None
        self.snapshot_daily_poc  = None

        # Live trade
        self.trade_direction = None
        self.trade_entry     = None
        self.trade_sl        = None
        self.trade_tp        = None
        self.trade_entry_dt  = None

        # Output
        self.trades : List[TPOTrade] = []
        self.events : List[Dict[str, Any]] = []   # state-transition log

    # ── State transitions ────────────────────────────────────────────────────

    def _log(self, bar, msg):
        if self.verbose:
            print(f"  [{bar['datetime']}] state={STATE_NAMES[self.state]:<16}  {msg}")

    def _reset_to_watching(self, bar, reason):
        if self.verbose and self.state != 1:
            self._log(bar, f"reset → WATCHING ({reason})")
        self.state            = 1
        self.probe_direction  = None
        self.probe_extreme    = None
        self.probe_dt         = None
        self.delta_confirm_dt = None
        self.back_inside_dt   = None
        # Snapshots cleared
        self.snapshot_atr        = None
        self.snapshot_velocity   = None
        self.snapshot_probe_dist = None
        self.snapshot_weekly_vah = None
        self.snapshot_weekly_val = None
        self.snapshot_weekly_poc = None
        self.snapshot_daily_vah  = None
        self.snapshot_daily_val  = None
        self.snapshot_daily_poc  = None

    def _check_va_acceptance(self, bar):
        """If migrated VA now engulfs probe extreme → acceptance, reset."""
        if self.state not in (2, 3, 4) or self.probe_direction is None:
            return False
        if self.probe_direction == 'above':
            if (self.engine.weekly_vah is not None and
                self.engine.weekly_vah >= self.probe_extreme):
                self.counters['va_acceptance_resets'] += 1
                self._reset_to_watching(bar, "VA acceptance (probe engulfed)")
                return True
        else:
            if (self.engine.weekly_val is not None and
                self.engine.weekly_val <= self.probe_extreme):
                self.counters['va_acceptance_resets'] += 1
                self._reset_to_watching(bar, "VA acceptance (probe engulfed)")
                return True
        return False

    # ── Main per-bar update ──────────────────────────────────────────────────

    def run(self):
        n = len(self.df_1h)
        for i in range(n):
            bar = self.df_1h.iloc[i]
            self._update_bar(bar, i)
        return self.trades

    def _update_bar(self, bar, i):
        # 1. Feed engine
        bar_minutes = self._m_index.get(bar['datetime'])
        self.engine.add_bar(bar, bar_minutes)

        # 2. Initialize state from IDLE once weekly VA exists
        if self.state == 0:
            if self.engine.weekly_vah is not None and self.engine.weekly_val is not None:
                self.state = 1
            return

        # 3. If we have a live trade, manage it FIRST (exits supersede everything)
        if self.state == 5:
            self._manage_trade(bar)
            return

        # 4. Continuous VA acceptance check (resets state if triggered)
        if self._check_va_acceptance(bar):
            return  # state was reset; processing this bar is done

        # 5. Branch by state
        if self.state == 1:
            self._handle_watching(bar)
        elif self.state == 2:
            self._handle_probe_pending(bar, i)
        elif self.state == 3:
            self._handle_probe_confirmed(bar)
        elif self.state == 4:
            self._handle_retest_waiting(bar)

    # ── State 1: WATCHING ────────────────────────────────────────────────────

    def _handle_watching(self, bar):
        vah = self.engine.weekly_vah
        val = self.engine.weekly_val
        if vah is None or val is None:
            return

        c = bar['close']
        if c > vah:
            self.counters['closes_outside_va'] += 1
            self.state           = 2
            self.probe_direction = 'above'
            self.probe_extreme   = float(bar['high'])
            self.probe_dt        = bar['datetime']
            self._log(bar, f"close {c:.2f} > VAH {vah:.2f} → PROBE_PENDING (above)")
        elif c < val:
            self.counters['closes_outside_va'] += 1
            self.state           = 2
            self.probe_direction = 'below'
            self.probe_extreme   = float(bar['low'])
            self.probe_dt        = bar['datetime']
            self._log(bar, f"close {c:.2f} < VAL {val:.2f} → PROBE_PENDING (below)")

    # ── State 2: PROBE_PENDING ───────────────────────────────────────────────

    def _handle_probe_pending(self, bar, i):
        vah = self.engine.weekly_vah
        val = self.engine.weekly_val

        # Update probe extreme (track furthest point, wick or body)
        if self.probe_direction == 'above':
            self.probe_extreme = max(self.probe_extreme, float(bar['high']))
        else:
            self.probe_extreme = min(self.probe_extreme, float(bar['low']))

        # If close back inside without confirmation → reset
        c = bar['close']
        still_outside = (c > vah) if self.probe_direction == 'above' else (c < val)
        if not still_outside:
            self.counters['closed_back_no_conf'] += 1
            self._reset_to_watching(bar, "closed back inside without confirmation")
            return

        # Check delta divergence on this bar
        delta = bar.get('delta', 0.0)
        if pd.isna(delta):
            delta = 0.0
        green = c > bar['open']
        red   = c < bar['open']

        is_divergence = False
        if self.probe_direction == 'above' and green and delta < 0:
            is_divergence = True
        elif self.probe_direction == 'below' and red and delta > 0:
            is_divergence = True
        if not is_divergence:
            return

        self.counters['delta_divergences'] += 1

        # Quality filters
        atr = bar['atr']
        if not np.isfinite(atr) or atr <= 0:
            return

        if self.probe_direction == 'above':
            distance = self.probe_extreme - vah
        else:
            distance = val - self.probe_extreme
        distance_atr = distance / atr

        # Filter 1: minimum distance
        if distance_atr < self.min_probe_dist_atr:
            self.counters['failed_min_distance'] += 1
            self._log(bar, f"divergence rejected: distance/ATR={distance_atr:.3f} < min")
            return
        self.counters['passed_min_distance'] += 1

        # Filter 2: velocity score
        rolling_vol = bar['rolling_vol']
        bar_vol     = float(bar['volume']) if np.isfinite(bar['volume']) else 0.0
        if (not np.isfinite(rolling_vol) or rolling_vol <= 0 or
            bar_vol <= 0):
            self._log(bar, "divergence rejected: invalid volume normalization")
            return
        velocity = (bar['range'] / atr) / (bar_vol / rolling_vol)
        if velocity >= self.velocity_threshold:
            self.counters['failed_velocity'] += 1
            self._log(bar, f"divergence rejected: velocity={velocity:.3f} ≥ thresh "
                           f"(spike disqualifier)")
            return
        self.counters['passed_velocity'] += 1
        self.counters['probe_confirmed'] += 1

        # Passed all filters → snapshot + transition to PROBE_CONFIRMED
        self.snapshot_atr        = float(atr)
        self.snapshot_velocity   = float(velocity)
        self.snapshot_probe_dist = float(distance_atr)
        self.snapshot_weekly_vah = float(vah)
        self.snapshot_weekly_val = float(val)
        self.snapshot_weekly_poc = float(self.engine.weekly_poc)
        self.snapshot_daily_vah  = float(self.engine.daily_vah) if self.engine.daily_vah is not None else float('nan')
        self.snapshot_daily_val  = float(self.engine.daily_val) if self.engine.daily_val is not None else float('nan')
        self.snapshot_daily_poc  = float(self.engine.daily_poc) if self.engine.daily_poc is not None else float('nan')
        self.delta_confirm_dt    = bar['datetime']
        self.state               = 3
        self._log(bar, f"divergence CONFIRMED: dist={distance_atr:.2f}ATR, "
                       f"vel={velocity:.2f}, delta={delta:+.0f} → PROBE_CONFIRMED")

    # ── State 3: PROBE_CONFIRMED — wait for back inside ──────────────────────

    def _handle_probe_confirmed(self, bar):
        vah = self.engine.weekly_vah
        val = self.engine.weekly_val

        # Continue tracking probe extreme (used for SL)
        if self.probe_direction == 'above':
            self.probe_extreme = max(self.probe_extreme, float(bar['high']))
        else:
            self.probe_extreme = min(self.probe_extreme, float(bar['low']))

        c = bar['close']
        back_inside = (c <= vah) if self.probe_direction == 'above' else (c >= val)
        if not back_inside:
            return

        self.counters['back_inside_events'] += 1

        # Daily VA overlap gate at the moment of back-inside event
        gate_open = self.engine.daily_poc_in_weekly_va()
        if gate_open:
            self.counters['gate_passed'] += 1
        else:
            self.counters['gate_failed'] += 1
            if self.gate_enabled:
                # Pre-registered behavior: gate is enforced
                self._log(bar, "back inside but daily POC outside weekly VA → filtered")
                self._reset_to_watching(bar, "daily VA overlap gate failed")
                return
            # Diagnostic mode (gate_enabled=False): bypass the gate
            self._log(bar, "back inside, daily gate FAILED but BYPASSED (diagnostic)")

        # Entry path branches on entry_mode
        self.back_inside_dt = bar['datetime']

        if self.entry_mode == 'back_inside':
            # Direct entry at back-inside bar's close (diagnostic variant)
            self.counters['immediate_entries'] += 1
            direction = 'short' if self.probe_direction == 'above' else 'long'
            entry_price = float(bar['close'])
            self._enter_trade(bar, direction=direction, entry_price=entry_price)
        else:
            # Pre-registered: wait for retest at VAH/VAL boundary
            self.state = 4
            self.counters['retest_started'] += 1
            self._log(bar, "→ RETEST_WAITING")

    # ── State 4: RETEST_WAITING — wait for boundary touch, watch POC ─────────

    def _handle_retest_waiting(self, bar):
        vah = self.engine.weekly_vah
        val = self.engine.weekly_val
        poc = self.engine.weekly_poc

        # POC cancellation: any wick touch of POC ends the setup
        if poc is not None and bar['low'] <= poc <= bar['high']:
            self.counters['poc_cancellations'] += 1
            self._log(bar, f"POC {poc:.2f} touched before fill → cancel")
            self._reset_to_watching(bar, "POC touched before fill")
            return

        # EOD on holiday: don't enter new trades on non-full-session days
        if not is_full_session(bar['datetime']):
            return

        tick      = Config.TICK_SIZE
            # Note: float arithmetic, not exact tick, but well within tolerance
        prox_pts  = Config.RETEST_PROXIMITY_TICKS * tick

        if self.probe_direction == 'above':
            # Short fade: limit short at VAH; fill if bar high reaches VAH - 4 ticks
            if bar['high'] >= vah - prox_pts:
                self.counters['retest_filled'] += 1
                self._enter_trade(bar, direction='short', entry_price=float(vah))
        else:
            # Long fade: limit long at VAL; fill if bar low reaches VAL + 4 ticks
            if bar['low'] <= val + prox_pts:
                self.counters['retest_filled'] += 1
                self._enter_trade(bar, direction='long', entry_price=float(val))

    # ── Trade entry (State 4 → 5) ────────────────────────────────────────────

    def _enter_trade(self, bar, direction, entry_price):
        atr = self.snapshot_atr
        if direction == 'short':
            sl = self.probe_extreme + self.sl_atr_mult * atr
        else:
            sl = self.probe_extreme - self.sl_atr_mult * atr

        tp = self.snapshot_weekly_poc

        self.trade_direction = direction
        self.trade_entry     = entry_price
        self.trade_sl        = sl
        self.trade_tp        = tp
        self.trade_entry_dt  = bar['datetime']
        self.state           = 5
        self._log(bar, f"FILLED {direction.upper()} @ {entry_price:.2f}  "
                       f"SL {sl:.2f}  TP {tp:.2f}")

    # ── State 5: IN_TRADE — manage to exit ───────────────────────────────────

    def _manage_trade(self, bar):
        d  = self.trade_direction
        sl = self.trade_sl
        tp = self.trade_tp

        # Check SL and TP within this bar
        # Convention: if both could hit in the same bar, assume SL hits first
        # (worst-case fill — academically conservative; pre-registered choice)
        if d == 'short':
            sl_hit = bar['high'] >= sl
            tp_hit = bar['low']  <= tp
            if sl_hit:
                self._close_trade(bar, sl, 'SL')
                return
            if tp_hit:
                self._close_trade(bar, tp, 'TP')
                return
        else:
            sl_hit = bar['low']  <= sl
            tp_hit = bar['high'] >= tp
            if sl_hit:
                self._close_trade(bar, sl, 'SL')
                return
            if tp_hit:
                self._close_trade(bar, tp, 'TP')
                return

        # EOD flatten check (close at THIS bar's close if it's the EOD bar)
        if (bar['datetime'].hour == Config.EOD_FLATTEN_HOUR and
            bar['datetime'].minute == 0):
            self._close_trade(bar, float(bar['close']), 'EOD')

    def _close_trade(self, bar, exit_price, reason):
        d           = self.trade_direction
        entry       = self.trade_entry
        sl          = self.trade_sl
        atr         = self.snapshot_atr

        if d == 'long':
            pnl_pts = exit_price - entry
        else:
            pnl_pts = entry - exit_price
        pnl_dollars = pnl_pts * Config.POINT_VALUE * self.contracts

        # R-multiple: 1R = SL distance from entry
        r_dist  = abs(entry - sl)
        r_mult  = pnl_pts / r_dist if r_dist > 0 else 0.0

        if self.engine.weekly_reset_dt is not None:
            hrs_into_wk = (self.trade_entry_dt - self.engine.weekly_reset_dt).total_seconds() / 3600.0
        else:
            hrs_into_wk = float('nan')

        trade = TPOTrade(
            entry_dt           = self.trade_entry_dt,
            direction          = d,
            entry_price        = float(entry),
            sl                 = float(sl),
            tp_poc             = float(self.trade_tp),
            exit_dt            = bar['datetime'],
            exit_price         = float(exit_price),
            exit_reason        = reason,
            pnl_points         = float(pnl_pts),
            pnl_dollars        = float(pnl_dollars),
            contracts          = self.contracts,
            atr_at_entry       = float(atr),
            probe_extreme      = float(self.probe_extreme),
            probe_distance_atr = float(self.snapshot_probe_dist),
            velocity_score     = float(self.snapshot_velocity),
            weekly_vah_at_entry= float(self.snapshot_weekly_vah),
            weekly_val_at_entry= float(self.snapshot_weekly_val),
            weekly_poc_at_entry= float(self.snapshot_weekly_poc),
            daily_vah_at_entry = float(self.snapshot_daily_vah),
            daily_val_at_entry = float(self.snapshot_daily_val),
            daily_poc_at_entry = float(self.snapshot_daily_poc),
            hrs_into_week      = float(hrs_into_wk),
            probe_dt           = self.probe_dt,
            delta_confirm_dt   = self.delta_confirm_dt,
            back_inside_dt     = self.back_inside_dt,
            r_multiple         = float(r_mult),
        )
        self.trades.append(trade)
        self.counters['trades_closed'] += 1
        self._log(bar, f"CLOSED  {reason}  @ {exit_price:.2f}  pnl={pnl_pts:+.2f}pts  "
                       f"R={r_mult:+.2f}")

        # Reset
        self.trade_direction = None
        self.trade_entry     = None
        self.trade_sl        = None
        self.trade_tp        = None
        self.trade_entry_dt  = None
        self._reset_to_watching(bar, "trade closed")


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience export
# ═══════════════════════════════════════════════════════════════════════════════

def trades_to_dataframe(trades: List[TPOTrade]) -> pd.DataFrame:
    """Convert a list of TPOTrade objects to a flat DataFrame for export."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])
