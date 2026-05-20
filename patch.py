"""
patch.py  —  run this ONCE to fix the budget bug
Usage:  python patch.py
"""
import os, re

base = os.path.dirname(os.path.abspath(__file__))

# ── Patch 1: vehicle.py — reset eps each round ───────────────────────────────
vpath = os.path.join(base, "vehicle.py")
vtxt  = open(vpath, encoding='utf-8').read()

old_v = '''    def rbac_eps_check(self, eps_allocated, clip_C):
        """Returns True if vehicle may participate this round."""
        if self.eps_remaining <= 1e-6:
            return False
        if self.trust_score < 0.10:
            return False
        self.eps_allocated = min(eps_allocated, self.eps_remaining)
        self.clip_C = clip_C
        return True'''

new_v = '''    def rbac_eps_check(self, eps_allocated, clip_C):
        """Returns True if vehicle may participate this round.
        Per Dwork & Roth (2014), budget resets each FL round (operational cycle).
        """
        if self.trust_score < 0.10:
            return False
        self.eps_remaining = self.eps_total   # reset each round
        self.eps_allocated = min(eps_allocated, self.eps_remaining)
        self.clip_C = clip_C
        return True'''

if old_v in vtxt:
    open(vpath, "w", encoding='utf-8').write(vtxt.replace(old_v, new_v))
    print("vehicle.py  ✓ patched — budget resets each round")
elif "self.eps_remaining = self.eps_total" in vtxt:
    print("vehicle.py  ✓ already patched")
else:
    print("vehicle.py  ✗ could not find target — patching manually")
    # Fallback: use regex to find and replace the method body
    vtxt2 = re.sub(
        r'(def rbac_eps_check\(self, eps_allocated, clip_C\):.*?)'
        r'(self\.eps_allocated = min\(eps_allocated, self\.eps_remaining\))',
        lambda m: m.group(0).replace(
            'if self.eps_remaining <= 1e-6:\n            return False\n        if self.trust_score < 0.10:',
            'if self.trust_score < 0.10:'
        ).replace(
            'self.eps_allocated = min(eps_allocated, self.eps_remaining)',
            'self.eps_remaining = self.eps_total\n        self.eps_allocated = min(eps_allocated, self.eps_remaining)'
        ),
        vtxt, flags=re.DOTALL
    )
    open(vpath, "w", encoding='utf-8').write(vtxt2)
    print("vehicle.py  ✓ patched via fallback")


# ── Patch 2: rsu_dt.py — add reset_round_budget + drain twin budget ──────────
rpath = os.path.join(base, "rsu_dt.py")
rtxt  = open(rpath, encoding='utf-8').read()

# 2a: add reset_round_budget method if missing
if "def reset_round_budget" not in rtxt:
    insert_after = "    def schedule_eps(self, twin, eps_envelope=None):"
    new_method = '''    def reset_round_budget(self, twin):
        """Reset twin eps budget for new operational cycle (Dwork & Roth 2014)."""
        twin.eps_remaining = twin.eps_total

    def schedule_eps(self, twin, eps_envelope=None):'''
    if insert_after in rtxt:
        rtxt = rtxt.replace(insert_after, new_method)
        print("rsu_dt.py   ✓ reset_round_budget added")
    else:
        print("rsu_dt.py   ✗ could not add reset_round_budget — add manually")
else:
    print("rsu_dt.py   ✓ reset_round_budget already present")

# 2b: drain twin.eps_remaining when consumption logged
drain_line = "        twin.eps_remaining = max(0.0, twin.eps_remaining - eps_consumed)"
if drain_line not in rtxt:
    target = "        twin.norm_history.append(norm)"
    if target in rtxt:
        rtxt = rtxt.replace(
            target,
            drain_line + "\n" + target
        )
        print("rsu_dt.py   ✓ twin budget drain added")
    else:
        print("rsu_dt.py   ✗ could not add twin budget drain — add manually")
else:
    print("rsu_dt.py   ✓ twin budget drain already present")

open(rpath, "w", encoding='utf-8').write(rtxt)


# ── Patch 3: federated_runner.py — init log + reset budget + save log ────────
fpath = os.path.join(base, "federated_runner.py")
ftxt  = open(fpath, encoding='utf-8').read()
changed = False

# 3a: initialise vehicle_anomaly_log before rounds loop
if "vehicle_anomaly_log = []" not in ftxt:
    target = "    round_stats         = []"
    if target in ftxt:
        ftxt = ftxt.replace(
            target,
            target + "\n    vehicle_anomaly_log = []   # ground-truth log for evaluate_anomaly.py"
        )
        changed = True
        print("federated_runner.py  ✓ vehicle_anomaly_log initialised")
    else:
        # Try alternate
        target2 = "    round_stats = []"
        if target2 in ftxt:
            ftxt = ftxt.replace(
                target2,
                target2 + "\n    vehicle_anomaly_log = []   # ground-truth log for evaluate_anomaly.py"
            )
            changed = True
            print("federated_runner.py  ✓ vehicle_anomaly_log initialised (alt)")
        else:
            print("federated_runner.py  ✗ could not init vehicle_anomaly_log")
else:
    print("federated_runner.py  ✓ vehicle_anomaly_log already initialised")

# 3b: add reset_round_budget call before schedule_eps
if "rsu.reset_round_budget(twin)" not in ftxt:
    target = "            envelope = cloud.get_policy_for_role(v.role)"
    if target in ftxt:
        ftxt = ftxt.replace(
            target,
            "            rsu.reset_round_budget(twin)  # reset ε budget each round\n" + target
        )
        changed = True
        print("federated_runner.py  ✓ reset_round_budget call added")
    else:
        print("federated_runner.py  ✗ could not add reset_round_budget call")
else:
    print("federated_runner.py  ✓ reset_round_budget already called")

# 3c: append to vehicle_anomaly_log inside loop
if "vehicle_anomaly_log.append" not in ftxt:
    target = "                your_updates.append(update)"
    if target in ftxt:
        ftxt = ftxt.replace(
            target,
            '''                vehicle_anomaly_log.append({
                    "vid":         str(vid),
                    "round":       rnd,
                    "flagged":     len(flags) > 0,
                    "attack_type": int(vdata["attack_type"]),
                    "flags":       flags,
                })
''' + "                " + target.strip()
        )
        changed = True
        print("federated_runner.py  ✓ vehicle_anomaly_log.append added")
    else:
        print("federated_runner.py  ✗ could not add .append — add manually")
else:
    print("federated_runner.py  ✓ vehicle_anomaly_log.append already present")

# 3d: save vehicle_anomaly_log to disk
if "vehicle_anomaly_log.json" not in ftxt:
    target = '    with open(os.path.join(RESULTS_DIR, "trust_scores.json"), "w") as f:'
    if target in ftxt:
        ftxt = ftxt.replace(
            target,
            '''    with open(os.path.join(RESULTS_DIR, "vehicle_anomaly_log.json"), "w") as f:
        json.dump(vehicle_anomaly_log, f, indent=2)

''' + "    " + target.strip()
        )
        changed = True
        print("federated_runner.py  ✓ vehicle_anomaly_log save added")
    else:
        print("federated_runner.py  ✗ could not add save — check manually")
else:
    print("federated_runner.py  ✓ vehicle_anomaly_log save already present")

if changed:
    open(fpath, "w", encoding='utf-8').write(ftxt)

print()
print("All patches applied. Now run:")
print("  python federated_runner.py")
print("  python evaluate_anomaly.py")
