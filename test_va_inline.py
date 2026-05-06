"""
Inline VA algorithm test — no pandas/pytz needed. Mirrors the exact
algorithm from TPOEngine._compute_va so we can verify correctness in
this constrained sandbox where pandas can't import.
"""

VA_THRESHOLD = 0.70

def compute_va(profile, tick_size=0.25):
    if not profile:
        return (None, None, None)
    total_vol = sum(profile.values())
    if total_vol <= 0:
        return (None, None, None)

    poc_tick  = max(profile, key=lambda t: (profile[t], -t))
    poc_price = poc_tick * tick_size

    target_vol = VA_THRESHOLD * total_vol
    captured   = profile[poc_tick]
    low_tick   = poc_tick
    high_tick  = poc_tick

    min_tick  = min(profile.keys())
    max_tick  = max(profile.keys())

    if min_tick == max_tick:
        return (poc_price, poc_price, poc_price)

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

    return (high_tick * tick_size, low_tick * tick_size, poc_price)


# Test 1: single tick collapses
v = compute_va({4000: 100.0})
assert v == (1000.0, 1000.0, 1000.0), v
print("Test 1 (single tick)         OK", v)

# Test 2: symmetric triangle — POC at center
profile = {4000: 10, 4001: 30, 4002: 60, 4003: 100, 4004: 60, 4005: 30, 4006: 10}
v = compute_va(profile)
print("Test 2 (symmetric triangle)  OK", v)
assert v == (1001.0, 1000.5, 1000.75), v

# Test 3: right-skewed (POC at low boundary)
profile = {4000: 200, 4001: 100, 4002: 50, 4003: 25, 4004: 12, 4005: 6, 4006: 3}
v = compute_va(profile)
print("Test 3 (right-skewed)        OK", v)
assert v == (1000.25, 1000.0, 1000.0), v

# Test 4: empty
v = compute_va({})
assert v == (None, None, None), v
print("Test 4 (empty)               OK", v)

# Test 5: random — captured fraction always ≥70%
import random
random.seed(42)
for trial in range(50):
    n_ticks = random.randint(5, 50)
    profile = {4000 + i: random.uniform(1, 100) for i in range(n_ticks)}
    vah, val, poc = compute_va(profile)
    total = sum(profile.values())
    captured = sum(v for t, v in profile.items()
                   if val/0.25 <= t <= vah/0.25)
    pct = captured / total * 100
    assert pct >= 69.99, f"Trial {trial}: only {pct:.2f}% captured"
print(f"Test 5 (50 random profiles)  OK — all ≥70% capture")

# Test 6: minute-bar volume distribution math
def add_minute(profile, low, high, vol, tick_size=0.25):
    low_tick  = int(round(low  / tick_size))
    high_tick = int(round(high / tick_size))
    n_ticks   = high_tick - low_tick + 1
    per       = vol / n_ticks
    for t in range(low_tick, high_tick + 1):
        profile[t] = profile.get(t, 0.0) + per

p = {}
add_minute(p, 1000.00, 1001.00, 1000.0)
# Should be 5 ticks (4000..4004) with 200 each
for t in [4000, 4001, 4002, 4003, 4004]:
    assert abs(p[t] - 200.0) < 1e-9, p
print("Test 6 (1M volume distrib)   OK — 1000 vol / 5 ticks → 200/tick")

# Test 7: POC tie-break determinism (lowest tick wins)
# tick-break rule: max(profile, key=lambda t: (profile[t], -t))
# Two ticks with same volume → -t means LARGER negative (more negative) is "less"
# So when ties on volume, it picks the one with smaller -t, i.e. LARGER t. Wait.
# max((v, -t)) → tie-break: prefer larger -t = smaller t.
# So lowest tick wins on tie. Verify:
p = {4000: 100, 4005: 100, 4010: 50}
v = compute_va(p)
assert v[2] == 4000 * 0.25, f"POC should be lowest-tick winner: 1000.00, got {v[2]}"
print(f"Test 7 (POC tie-break)       OK — lowest tick wins on volume tie")

print("\nAll inline VA tests passed.")
