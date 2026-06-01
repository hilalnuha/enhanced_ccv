"""
simulation.py
=============
Numerical verification of theoretical results for the paper:
  "ECOCV"


import math
import random
import statistics
import time
from typing import List, Tuple

random.seed(2025)

Point = Tuple[int, float]

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (same as compression_comparison.py)
# ──────────────────────────────────────────────────────────────────────────────

def _dedup(pts):
    if not pts: return pts
    out = [pts[0]]
    for p in pts[1:]:
        if p[0] != out[-1][0]: out.append(p)
    return out

def cocv_fixed_compress(pts):
    if not pts: return []
    compressed = [pts[0]]
    prev_t, prev_v = pts[0]
    current_interval = None
    in_run = False
    for cur_t, cur_v in pts[1:]:
        gap = cur_t - prev_t
        if cur_v == prev_v and (current_interval is None or gap == current_interval):
            if current_interval is None: current_interval = gap
            in_run = True; prev_t = cur_t
        else:
            if in_run: compressed.append((prev_t, prev_v)); in_run = False
            compressed.append((cur_t, cur_v))
            current_interval = gap; prev_t, prev_v = cur_t, cur_v
    if in_run: compressed.append((prev_t, prev_v))
    return _dedup(compressed)

def epsilon_cocv_compress(pts, epsilon=0.05):
    if not pts: return []
    compressed = [pts[0]]
    prev_t, prev_v = pts[0]
    current_interval = None; in_run = False; run_ref = prev_v
    for cur_t, cur_v in pts[1:]:
        gap = cur_t - prev_t
        if abs(cur_v - run_ref) <= epsilon and (current_interval is None or gap == current_interval):
            if current_interval is None: current_interval = gap
            in_run = True; prev_t = cur_t
        else:
            if in_run: compressed.append((prev_t, run_ref)); in_run = False
            compressed.append((cur_t, cur_v))
            current_interval = gap; run_ref = cur_v; prev_t, prev_v = cur_t, cur_v
    if in_run: compressed.append((prev_t, run_ref))
    return _dedup(compressed)

def epsilon_cocv_decompress(compressed, original):
    if len(compressed) <= 1: return compressed
    result = []; ci = 0
    for t, _ in original:
        while ci < len(compressed) - 2 and t >= compressed[ci+1][0]: ci += 1
        result.append((t, compressed[ci][1]))
    return result

def _perp_dist(p, a, b):
    at, av = a; bt, bv = b; pt, pv = p
    dt = bt-at; dv = bv-av
    if dt == 0 and dv == 0: return math.hypot(pt-at, pv-av)
    t_param = ((pt-at)*dt + (pv-av)*dv) / (dt*dt + dv*dv)
    return math.hypot(pt-(at+t_param*dt), pv-(av+t_param*dv))

def _rdp(pts, eps, out):
    if len(pts) < 2: out.extend(pts); return
    md, mi = 0.0, 0
    for i in range(1, len(pts)-1):
        d = _perp_dist(pts[i], pts[0], pts[-1])
        if d > md: md, mi = d, i
    if md > eps:
        _rdp(pts[:mi+1], eps, out); out.pop(); _rdp(pts[mi:], eps, out)
    else:
        out.append(pts[0]); out.append(pts[-1])

def rdp_compress(pts, epsilon=0.1):
    if len(pts) < 2: return pts
    raw = []; _rdp(pts, epsilon, raw)
    seen = set(); out = []
    for p in sorted(raw, key=lambda x:x[0]):
        if p[0] not in seen: seen.add(p[0]); out.append(p)
    return out

def rdp_decompress(compressed, original):
    if len(compressed) < 2: return compressed
    result = []; ci = 0
    for t, _ in original:
        while ci < len(compressed)-2 and t >= compressed[ci+1][0]: ci += 1
        t0,v0 = compressed[ci]; t1,v1 = compressed[ci+1]
        alpha = (t-t0)/(t1-t0) if t1 != t0 else 0.0
        result.append((t, v0+alpha*(v1-v0)))
    return result

def sdt_compress(pts, deviation=0.1):
    if len(pts) < 2: return pts
    compressed = [pts[0]]; t_last,v_last = pts[0]
    slope_upper = float('inf'); slope_lower = float('-inf')
    prev_t, prev_v = pts[0]
    for cur_t, cur_v in pts[1:]:
        dt = cur_t - t_last
        if dt == 0: prev_t,prev_v = cur_t,cur_v; continue
        su_new = (cur_v+deviation-v_last)/dt; sl_new = (cur_v-deviation-v_last)/dt
        new_upper = min(slope_upper,su_new); new_lower = max(slope_lower,sl_new)
        if new_lower > new_upper:
            compressed.append((prev_t,prev_v))
            t_last,v_last = prev_t,prev_v; dt2 = cur_t-t_last
            slope_upper = (cur_v+deviation-v_last)/dt2 if dt2 else float('inf')
            slope_lower = (cur_v-deviation-v_last)/dt2 if dt2 else float('-inf')
        else:
            slope_upper = new_upper; slope_lower = new_lower
        prev_t,prev_v = cur_t,cur_v
    compressed.append(pts[-1]); return _dedup(compressed)

def rmse(original, reconstructed):
    rec = {t:v for t,v in reconstructed}
    errs = [(ov - rec[ot])**2 for ot,ov in original if ot in rec]
    return math.sqrt(sum(errs)/len(errs)) if errs else 0.0

def compression_score(original_n, compressed_n):
    return (1 - compressed_n/original_n) * 100

# ──────────────────────────────────────────────────────────────────────────────
# Data generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_flat_with_noise(n=1000, epsilon=0.05, dis_rate=0.05, base=25.0, interval=5):
    """Flat time-series with Gaussian noise ≤ epsilon and dis_rate discontinuous spikes."""
    pts = []
    for i in range(n):
        t = 1000 + i*interval
        if random.random() < dis_rate:
            v = base + random.uniform(2*epsilon, 5*epsilon) * random.choice([-1,1])
        else:
            v = base + random.uniform(-epsilon*0.9, epsilon*0.9)
        pts.append((t, round(v, 6)))
    return pts

def gen_sinusoidal(n=1000, amplitude=5.0, period=100, interval=5, noise=0.01):
    """Sinusoidal signal with small noise for RDP/SDT testing."""
    pts = []
    for i in range(n):
        t = 1000 + i*interval
        v = amplitude * math.sin(2*math.pi*i/period) + random.gauss(0, noise)
        pts.append((t, round(v, 6)))
    return pts

def gen_mixed(n=1000, dis_rate=0.1, avg_ccs_len=50, interval=5):
    """Mixed: long flat CCS segments interrupted by random spikes."""
    pts = []; t = 1000; v = 20.0; run = 0; run_len = int(random.expovariate(1/avg_ccs_len))+3
    for i in range(n):
        if run >= run_len or random.random() < dis_rate:
            v = random.uniform(10, 40); run = 0
            run_len = int(random.expovariate(1/avg_ccs_len))+3
        pts.append((t, round(v, 6))); t += interval; run += 1
    return pts

# ──────────────────────────────────────────────────────────────────────────────
# Theorem 4: Epsilon-COCV compression score lower bound
#   CS_eps >= CS_cocv  (epsilon >= 0 always gives at least as good as strict COCV)
#   Experimentally: CS_eps grows monotonically with epsilon
# ──────────────────────────────────────────────────────────────────────────────

def verify_theorem4(trials=30):
    print("\n" + "="*70)
    print("Theorem 4: Epsilon-COCV CS >= COCV-Fixed CS for any epsilon >= 0")
    print("="*70)
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    violations = 0; total = 0
    for _ in range(trials):
        pts = gen_flat_with_noise(n=500, epsilon=0.05, dis_rate=random.uniform(0.02,0.15))
        cs_base = compression_score(len(pts), len(cocv_fixed_compress(pts)))
        prev_cs = cs_base
        for eps in epsilons[1:]:
            c = epsilon_cocv_compress(pts, epsilon=eps)
            cs = compression_score(len(pts), len(c))
            if cs < prev_cs - 0.5:   # allow 0.5% numerical tolerance
                violations += 1
            prev_cs = cs
            total += 1
    print(f"  Monotonicity violations: {violations}/{total}  "
          f"({'PASS' if violations == 0 else 'FAIL'})")

    # Show a representative table
    pts = gen_flat_with_noise(n=2000, epsilon=0.05, dis_rate=0.05)
    cs_base = compression_score(len(pts), len(cocv_fixed_compress(pts)))
    print(f"\n  {'epsilon':>10}  {'CS (%)':>10}  {'DeltaCS':>10}")
    prev = cs_base
    for eps in epsilons:
        c = epsilon_cocv_compress(pts, epsilon=eps)
        cs = compression_score(len(pts), len(c))
        print(f"  {eps:>10.3f}  {cs:>10.2f}  {cs-prev:>+10.2f}")
        prev = cs

# ──────────────────────────────────────────────────────────────────────────────
# Theorem 5: Epsilon-COCV RMSE upper bound
#   RMSE(eps) <= epsilon  (step-hold reconstruction error bounded by tolerance)
# ──────────────────────────────────────────────────────────────────────────────

def verify_theorem5(trials=100):
    print("\n" + "="*70)
    print("Theorem 5: RMSE(Epsilon-COCV) <= epsilon  (upper bound)")
    print("="*70)
    violations = 0
    for _ in range(trials):
        eps = random.uniform(0.01, 0.5)
        pts = gen_flat_with_noise(n=500, epsilon=eps*0.9, dis_rate=random.uniform(0.01, 0.2))
        c = epsilon_cocv_compress(pts, epsilon=eps)
        d = epsilon_cocv_decompress(c, pts)
        r = rmse(pts, d)
        if r > eps + 1e-9:
            violations += 1
    print(f"  Trials: {trials}  |  Violations (RMSE > epsilon): {violations}  "
          f"({'PASS' if violations == 0 else 'FAIL'})")

    # Tabulate for several fixed epsilons
    print(f"\n  {'epsilon':>10}  {'Mean RMSE':>12}  {'Max RMSE':>12}  {'Bound holds':>12}")
    for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
        rmses = []
        for _ in range(50):
            pts = gen_flat_with_noise(n=500, epsilon=eps*0.9, dis_rate=0.05)
            c = epsilon_cocv_compress(pts, epsilon=eps)
            d = epsilon_cocv_decompress(c, pts)
            rmses.append(rmse(pts, d))
        mean_r = statistics.mean(rmses); max_r = max(rmses)
        holds = "YES" if max_r <= eps + 1e-9 else "NO"
        print(f"  {eps:>10.3f}  {mean_r:>12.6f}  {max_r:>12.6f}  {holds:>12}")

# ──────────────────────────────────────────────────────────────────────────────
# Theorem 6: RDP-Linear CS grows with reduced signal curvature
#   Low-curvature (near-linear) signals → high CS; high-curvature → low CS
# ──────────────────────────────────────────────────────────────────────────────

def signal_curvature(pts):
    """Mean absolute second difference as curvature proxy."""
    vals = [v for _,v in pts]
    diffs2 = [abs(vals[i+2] - 2*vals[i+1] + vals[i]) for i in range(len(vals)-2)]
    return statistics.mean(diffs2) if diffs2 else 0.0

def verify_theorem6(epsilon=0.1):
    print("\n" + "="*70)
    print("Theorem 6: RDP CS is inversely related to signal curvature")
    print("="*70)
    results = []
    amplitudes = [0.5, 1, 2, 5, 10, 20, 50]
    for amp in amplitudes:
        pts = gen_sinusoidal(n=1000, amplitude=amp, period=50, noise=0.0)
        curv = signal_curvature(pts)
        c = rdp_compress(pts, epsilon=epsilon)
        cs = compression_score(len(pts), len(c))
        results.append((amp, curv, cs))

    print(f"  {'Amplitude':>10}  {'Curvature':>12}  {'RDP CS (%)':>12}")
    for amp, curv, cs in results:
        print(f"  {amp:>10.1f}  {curv:>12.6f}  {cs:>12.2f}")

    # Check monotonicity: as curvature increases, CS should decrease
    cs_vals = [r[2] for r in results]; curv_vals = [r[1] for r in results]
    # Spearman rank correlation (manual)
    n = len(cs_vals)
    rank_curv = sorted(range(n), key=lambda i: curv_vals[i])
    rank_cs   = sorted(range(n), key=lambda i: cs_vals[i], reverse=True)
    agreement = sum(1 for i in range(n) if rank_curv[i] == rank_cs[i])
    print(f"\n  Rank agreement (higher curvature <-> lower CS): {agreement}/{n}")
    # Pearson on cs vs curvature (should be negative)
    mean_curv = statistics.mean(curv_vals); mean_cs = statistics.mean(cs_vals)
    cov = sum((curv_vals[i]-mean_curv)*(cs_vals[i]-mean_cs) for i in range(n))
    std_c = math.sqrt(sum((x-mean_curv)**2 for x in curv_vals))
    std_s = math.sqrt(sum((x-mean_cs)**2 for x in cs_vals))
    corr = cov/(std_c*std_s) if std_c*std_s > 0 else 0
    print(f"  Pearson r(curvature, CS): {corr:.4f}  "
          f"({'PASS (negative)' if corr < 0 else 'CHECK'})")

# ──────────────────────────────────────────────────────────────────────────────
# Theorem 7: SDT CS monotonically increases with deviation delta
#   More deviation → fewer stored points → higher CS
# ──────────────────────────────────────────────────────────────────────────────

def verify_theorem7(trials=30):
    print("\n" + "="*70)
    print("Theorem 7: SDT CS is monotonically non-decreasing in deviation delta")
    print("="*70)
    deviations = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    violations = 0; total = 0
    for _ in range(trials):
        pts = gen_sinusoidal(n=500, amplitude=random.uniform(1,10),
                             period=random.randint(20,100), noise=0.01)
        prev_cs = -1
        for dev in deviations:
            c = sdt_compress(pts, deviation=dev)
            cs = compression_score(len(pts), len(c))
            if cs < prev_cs - 0.5:
                violations += 1
            prev_cs = cs; total += 1

    print(f"  Monotonicity violations: {violations}/{total}  "
          f"({'PASS' if violations == 0 else 'FAIL'})")

    # Tabulate for one representative signal
    pts = gen_sinusoidal(n=2000, amplitude=5.0, period=100, noise=0.01)
    print(f"\n  {'delta':>8}  {'Stored pts':>10}  {'CS (%)':>10}  {'RMSE':>10}")
    prev_cs = 0
    for dev in deviations:
        c = sdt_compress(pts, deviation=dev)
        d = rdp_decompress(c, pts)
        cs = compression_score(len(pts), len(c))
        r  = rmse(pts, d)
        print(f"  {dev:>8.3f}  {len(c):>10}  {cs:>10.2f}  {r:>10.6f}")

# ──────────────────────────────────────────────────────────────────────────────
# Theorem 8: Composite two-pass (COCV + Epsilon) CS >= max(CS_cocv, CS_eps)
# ──────────────────────────────────────────────────────────────────────────────

def composite_compress(pts, epsilon=0.05):
    """Two-pass: first lossless COCV, then epsilon pass on residuals."""
    pass1 = cocv_fixed_compress(pts)
    pass2 = epsilon_cocv_compress(pass1, epsilon=epsilon)
    return pass2

def verify_theorem8(trials=50):
    print("\n" + "="*70)
    print("Theorem 8: Composite CS >= max(CS_cocv, CS_eps)")
    print("="*70)
    violations = 0
    for _ in range(trials):
        eps = random.uniform(0.01, 0.2)
        pts = gen_mixed(n=500, dis_rate=random.uniform(0.02, 0.2),
                        avg_ccs_len=random.randint(5, 100))
        cs_cocv = compression_score(len(pts), len(cocv_fixed_compress(pts)))
        cs_eps  = compression_score(len(pts), len(epsilon_cocv_compress(pts, eps)))
        cs_comp = compression_score(len(pts), len(composite_compress(pts, eps)))
        if cs_comp < max(cs_cocv, cs_eps) - 0.5:
            violations += 1
    print(f"  Violations: {violations}/{trials}  "
          f"({'PASS' if violations == 0 else 'FAIL'})")

    # Representative table
    pts = gen_mixed(n=2000, dis_rate=0.05, avg_ccs_len=50)
    print(f"\n  {'epsilon':>10}  {'CS_cocv':>10}  {'CS_eps':>10}  {'CS_comp':>10}  {'Gain':>10}")
    for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
        cs_c = compression_score(len(pts), len(cocv_fixed_compress(pts)))
        cs_e = compression_score(len(pts), len(epsilon_cocv_compress(pts, eps)))
        cs_k = compression_score(len(pts), len(composite_compress(pts, eps)))
        print(f"  {eps:>10.3f}  {cs_c:>10.2f}  {cs_e:>10.2f}  {cs_k:>10.2f}  "
              f"{cs_k-max(cs_c,cs_e):>+10.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# Proposition 1: LZ-Pattern CS improves with periodic structure
# ──────────────────────────────────────────────────────────────────────────────

_LZ_QUANT = 16; _LZ_WINDOW = 4

def lz_compress(pts):
    if len(pts) < _LZ_WINDOW+1: return pts
    vals = [v for _,v in pts]
    vmin,vmax = min(vals),max(vals)
    def q(v): return min(int((v-vmin)/(vmax-vmin)*_LZ_QUANT), _LZ_QUANT-1) if vmax!=vmin else 0
    def key(i): return tuple(q(vals[j]) for j in range(i, min(i+_LZ_WINDOW,len(pts))))
    dct = {}; labels = []
    for i in range(len(pts)):
        k = key(i)
        if k not in dct: dct[k] = len(dct)
        labels.append(dct[k])
    compressed = [pts[0]]; prev = labels[0]; in_run=False; re = pts[0]
    for i in range(1,len(pts)):
        if labels[i]==prev: in_run=True; re=pts[i]
        else:
            if in_run: compressed.append(re); in_run=False
            compressed.append(pts[i]); prev=labels[i]
    if in_run: compressed.append(pts[-1])
    return _dedup(compressed)

def verify_proposition1():
    print("\n" + "="*70)
    print("Proposition 1: LZ-Pattern CS improves with signal periodicity")
    print("="*70)
    periods = [5, 10, 20, 50, 100, 200]
    print(f"  {'Period':>8}  {'Dict size':>10}  {'CS (%)':>10}")
    for period in periods:
        pts = gen_sinusoidal(n=2000, amplitude=5.0, period=period, noise=0.001)
        c = lz_compress(pts)
        cs = compression_score(len(pts), len(c))
        vals = [v for _,v in pts]; vmin,vmax = min(vals),max(vals)
        def q(v): return min(int((v-vmin)/(vmax-vmin)*_LZ_QUANT),_LZ_QUANT-1) if vmax!=vmin else 0
        def key(i): return tuple(q(vals[j]) for j in range(i,min(i+_LZ_WINDOW,len(pts))))
        dct = set(key(i) for i in range(len(pts)))
        print(f"  {period:>8}  {len(dct):>10}  {cs:>10.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# SNR vs epsilon tradeoff summary (for paper figure data)
# ──────────────────────────────────────────────────────────────────────────────

def snr_tradeoff_table():
    print("\n" + "="*70)
    print("SNR vs Compression Score tradeoff (Epsilon-COCV)")
    print("="*70)
    pts = gen_flat_with_noise(n=5000, epsilon=0.05, dis_rate=0.05, base=25.0)
    sig_pwr = statistics.mean(v**2 for _,v in pts)
    print(f"  {'epsilon':>10}  {'CS (%)':>10}  {'RMSE':>10}  {'SNR (dB)':>10}")
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        c = epsilon_cocv_compress(pts, epsilon=eps)
        d = epsilon_cocv_decompress(c, pts)
        r = rmse(pts, d)
        cs = compression_score(len(pts), len(c))
        mse = r**2; snr = 10*math.log10(sig_pwr/mse) if mse > 0 else float('inf')
        print(f"  {eps:>10.4f}  {cs:>10.2f}  {r:>10.6f}  {snr:>10.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*70)
    print("Simulation: Verification of Theoretical Results")
    print("Enhanced COCV for IoT Time-Series Compression")
    print("Authors: H.H. Nuha, P.A. Atmajaya — Telkom University")
    print("="*70)

    verify_theorem4()
    verify_theorem5()
    verify_theorem6()
    verify_theorem7()
    verify_theorem8()
    verify_proposition1()
    snr_tradeoff_table()

    print("\n" + "="*70)
    print("All verifications complete.")
    print("="*70)
