"""
fix_and_run.py  —  run this instead of federated_runner.py
This script fixes the files in-place AND immediately shows proof they are fixed.
Usage:  python fix_and_run.py
"""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

def read(fname):
    return open(os.path.join(BASE, fname), encoding='utf-8').read()

def write(fname, content):
    open(os.path.join(BASE, fname), 'w', encoding='utf-8').write(content)

print("=" * 55)
print("  STEP 1: Verifying and fixing files")
print("=" * 55)

# ── Fix vehicle.py ────────────────────────────────────────────────────────────
vtxt = read('vehicle.py')
if 'eps_remaining <= 1e-6' in vtxt:
    vtxt = vtxt.replace(
        '        if self.eps_remaining <= 1e-6:\n            return False\n        if self.trust_score < 0.10:',
        '        if self.trust_score < 0.10:'
    )
    vtxt = vtxt.replace(
        '        self.eps_allocated = min(eps_allocated, self.eps_remaining)\n        self.clip_C = clip_C\n        return True',
        '        self.eps_remaining = self.eps_total  # reset each round\n        self.eps_allocated = min(eps_allocated, self.eps_remaining)\n        self.clip_C = clip_C\n        return True'
    )
    write('vehicle.py', vtxt)
    print("vehicle.py       FIXED — budget now resets each round")
elif 'self.eps_remaining = self.eps_total' in vtxt:
    print("vehicle.py       already correct")
else:
    print("vehicle.py       WARNING: unexpected format, attempting force-fix")
    # nuclear option: rewrite the whole method
    import re
    vtxt = re.sub(
        r'def rbac_eps_check\(self.*?return True',
        '''def rbac_eps_check(self, eps_allocated, clip_C):
        if self.trust_score < 0.10:
            return False
        self.eps_remaining = self.eps_total
        self.eps_allocated = min(eps_allocated, self.eps_remaining)
        self.clip_C = clip_C
        return True''',
        vtxt, flags=re.DOTALL
    )
    write('vehicle.py', vtxt)
    print("vehicle.py       force-fixed")

# ── Fix rsu_dt.py ─────────────────────────────────────────────────────────────
rtxt = read('rsu_dt.py')
changed_r = False

if 'def reset_round_budget' not in rtxt:
    # Insert before schedule_eps
    rtxt = rtxt.replace(
        '    def schedule_eps(self, twin, eps_envelope=None):',
        '''    def reset_round_budget(self, twin):
        """Reset ε budget each operational cycle (Dwork & Roth 2014)."""
        twin.eps_remaining = twin.eps_total

    def schedule_eps(self, twin, eps_envelope=None):'''
    )
    changed_r = True
    print("rsu_dt.py        FIXED — reset_round_budget added")
else:
    print("rsu_dt.py        reset_round_budget already present")

drain = '        twin.eps_remaining = max(0.0, twin.eps_remaining - eps_consumed)'
if drain not in rtxt:
    rtxt = rtxt.replace(
        '        twin.norm_history.append(norm)',
        drain + '\n        twin.norm_history.append(norm)'
    )
    changed_r = True
    print("rsu_dt.py        FIXED — twin budget drain added")
else:
    print("rsu_dt.py        twin budget drain already present")

if changed_r:
    write('rsu_dt.py', rtxt)

# ── Fix federated_runner.py ───────────────────────────────────────────────────
ftxt = read('federated_runner.py')
changed_f = False

if 'vehicle_anomaly_log = []' not in ftxt:
    for anchor in ['    round_stats         = []', '    round_stats = []', '    round_stats=[]']:
        if anchor in ftxt:
            ftxt = ftxt.replace(anchor, anchor + '\n    vehicle_anomaly_log = []')
            changed_f = True
            print("federated_runner.py  FIXED — vehicle_anomaly_log initialised")
            break
    else:
        print("federated_runner.py  WARNING: could not find round_stats line")

if 'rsu.reset_round_budget(twin)' not in ftxt:
    for anchor in [
        '            envelope = cloud.get_policy_for_role(v.role)',
        '            envelope = cloud.get_policy_for_role(v.role)\n',
    ]:
        if anchor in ftxt:
            ftxt = ftxt.replace(anchor, '            rsu.reset_round_budget(twin)\n' + anchor)
            changed_f = True
            print("federated_runner.py  FIXED — reset_round_budget call added")
            break
    else:
        print("federated_runner.py  WARNING: could not find envelope line")

if 'vehicle_anomaly_log.append' not in ftxt:
    for anchor in [
        '                your_updates.append(update)\n',
        '                your_updates.append(update)',
    ]:
        if anchor in ftxt:
            append_block = '''                vehicle_anomaly_log.append({
                    "vid":         str(vid),
                    "round":       rnd,
                    "flagged":     len(flags) > 0,
                    "attack_type": int(vdata["attack_type"]),
                    "flags":       flags,
                })
'''
            ftxt = ftxt.replace(anchor, append_block + anchor)
            changed_f = True
            print("federated_runner.py  FIXED — anomaly log append added")
            break
    else:
        print("federated_runner.py  WARNING: could not find your_updates.append")

if 'vehicle_anomaly_log.json' not in ftxt:
    for anchor in [
        '    with open(os.path.join(RESULTS_DIR, "trust_scores.json")',
        "    with open(os.path.join(RESULTS_DIR, 'trust_scores.json')",
    ]:
        if anchor in ftxt:
            save_block = '''    with open(os.path.join(RESULTS_DIR, "vehicle_anomaly_log.json"), "w", encoding="utf-8") as f:
        import json as _json; _json.dump(vehicle_anomaly_log, f, indent=2)

'''
            ftxt = ftxt.replace(anchor, save_block + anchor)
            changed_f = True
            print("federated_runner.py  FIXED — anomaly log save added")
            break
    else:
        print("federated_runner.py  WARNING: could not find trust_scores save")

if changed_f:
    write('federated_runner.py', ftxt)

# ── Verify ────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  STEP 2: Verification")
print("=" * 55)

vtxt2 = read('vehicle.py')
rtxt2 = read('rsu_dt.py')
ftxt2 = read('federated_runner.py')

checks = [
    ("vehicle.py has budget reset",        'self.eps_remaining = self.eps_total' in vtxt2),
    ("vehicle.py no old depletion check",  'eps_remaining <= 1e-6' not in vtxt2),
    ("rsu_dt.py has reset_round_budget",   'def reset_round_budget' in rtxt2),
    ("rsu_dt.py has twin drain",           'twin.eps_remaining = max(0.0,' in rtxt2),
    ("runner has log init",                'vehicle_anomaly_log = []' in ftxt2),
    ("runner calls reset_round_budget",    'rsu.reset_round_budget(twin)' in ftxt2),
    ("runner has log append",              'vehicle_anomaly_log.append' in ftxt2),
    ("runner saves log to disk",           'vehicle_anomaly_log.json' in ftxt2),
]

all_ok = True
for desc, ok in checks:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {desc}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("All checks passed. Running federated_runner.py now...")
    print("=" * 55)
    print()
    # Run it
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, 'federated_runner.py')],
        cwd=BASE
    )
    if result.returncode == 0:
        print()
        print("federated_runner.py completed successfully.")
        print("Now run:  python evaluate_anomaly.py")
    else:
        print("federated_runner.py exited with error code", result.returncode)
else:
    print("Some checks FAILED. Please share the output above.")
