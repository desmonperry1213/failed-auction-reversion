# ═══════════════════════════════════════════════════════════════════════════════
# TPO ENGINE — Visual validation runner
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase 1 acceptance: visually validate the engine on at least 5 known weeks
# before moving to Phase 2 (pre-registration parameter analysis).
#
# Save this file alongside strategy.py and tpo_engine.py at:
#     C:\Trading\InitialBalanceAuctionFade\test_tpo.py
#
# Run from that folder:
#     python test_tpo.py
#
# Edit WEEKS_TO_VIEW below to step through the weeks you want to inspect.
# Five suggested defaults are pre-loaded — pick weeks you already know well
# from your IB-era strategy testing so you have a mental model for what the
# weekly POC and VA should look like.
#
# Validation checklist for each week:
#   [ ] POC visually corresponds to the densest zone of the heatmap
#   [ ] VAH and VAL bracket roughly the central 70% of the histogram mass
#   [ ] Weekly levels stabilize as the week progresses (less drift later)
#   [ ] Daily levels move noticeably more than weekly levels
#   [ ] Overlap step plot toggles when daily POC crosses weekly VA boundary
#   [ ] Final weekly POC lines up with what TradingView volume profile shows
#       (use the Session Volume Profile indicator on TV, set to weekly,
#        timezone US/Central — back-adjusted contract will offset prices but
#        the SHAPE and POC location relative to the week's price action
#        should match)
# ═══════════════════════════════════════════════════════════════════════════════

import os

from strategy import load_and_prepare_data
from tpo_engine import view_tpo_week


# ── Edit this list to step through the weeks you want to inspect ──────────────
# Format: ("YYYY-MM-DD start", "YYYY-MM-DD end")
# Use Sunday → following Saturday for full Globex weeks.
WEEKS_TO_VIEW = [
    ("2026-04-26", "2026-05-02"),  # Last full ES week before now (the Issue 2 week)
    ("2024-09-15", "2024-09-21"),  # FOMC week September 2024 (50bp cut)
    ("2024-08-04", "2024-08-10"),  # Vol-spike week (yen carry unwind)
    ("2023-03-12", "2023-03-18"),  # SVB collapse week
    ("2020-03-15", "2020-03-21"),  # COVID crash week — trend, not balance
]

SAVE_DIR = r"C:\Trading\InitialBalanceAuctionFade\results\tpo_validation"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Loading data...")
    df_1h, df_1m = load_and_prepare_data()
    print(f"Loaded df_1h: {len(df_1h):,} bars, df_1m: {len(df_1m):,} bars")

    for i, (start, end) in enumerate(WEEKS_TO_VIEW, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(WEEKS_TO_VIEW)}] Week: {start}  →  {end}")
        print(f"{'='*70}")
        save_path = os.path.join(SAVE_DIR, f"tpo_week_{start}_to_{end}.png")
        view_tpo_week(df_1h, df_1m, start, end, save_path=save_path, show=True)

    print(f"\n{'='*70}")
    print(f"  Validation runs complete. PNGs saved to:")
    print(f"  {SAVE_DIR}")
    print(f"{'='*70}\n")
    print("If all 5 weeks look correct, commit Config.TPO_ENABLED to True")
    print("in strategy.py and proceed to Phase 2 (pre-registration analysis).")


if __name__ == "__main__":
    main()
