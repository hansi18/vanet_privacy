"""
federated_runner.py
Main orchestrator. Runs your model (DT-RBAC-FL-ADP) and all 3 baselines
through 20 FL rounds on the VeReMi dataset, then evaluates and saves results.

Usage:
    python federated_runner.py
"""

import numpy as np
import torch
import random
import json
import os
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)
from sklearn.model_selection import train_test_split as tts
import warnings
warnings.filterwarnings("ignore")

from config import *
from preprocessing import load_veremi, engineer_features, prepare_vehicle_datasets, FEATURE_COLS
from models import AttackDetector
from vehicle import Vehicle, ROLE_TRUST, ROLE_EPS_RANGE
from rsu_dt import RSUDT
from cloud import CloudDT
from baselines import FLDPBaseline, DTFLBaseline, AdaptiveDPFLBaseline

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model_or_fn, X_test, y_test):
    if callable(model_or_fn) and not hasattr(model_or_fn, 'predict'):
        preds = model_or_fn(X_test)
    else:
        preds = model_or_fn.predict(X_test)
    preds = np.array(preds)
    return {
        "accuracy":  float(accuracy_score(y_test, preds)),
        "f1":        float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall":    float(recall_score(y_test, preds, zero_division=0)),
    }


def predict_with_model(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    return out.argmax(dim=1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DT-RBAC-FL-ADP vs Baselines — VeReMi Dataset")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_veremi(DATA_PATH, max_vehicles=MAX_VEHICLES, verbose=True)
    df = engineer_features(df, verbose=True)

    train_df, test_df = tts(df, test_size=0.20, random_state=SEED,
                            stratify=df["label"])

    vehicle_data_train, scaler = prepare_vehicle_datasets(train_df.copy())
    vehicle_data_test, _       = prepare_vehicle_datasets(test_df.copy(),
                                                          scaler=scaler,
                                                          fit_scaler=False)

    X_test_all = scaler.transform(test_df[FEATURE_COLS].values).astype(np.float32)
    y_test_all = test_df["label"].values.astype(np.int64)

    # Class weights
    n_legit  = int((train_df["label"] == 0).sum())
    n_attack = int((train_df["label"] == 1).sum())
    total    = n_legit + n_attack
    class_weights = [float(np.clip(total / (2 * n_legit  + 1e-9), 0.1, 10.0)),
                     float(np.clip(total / (2 * n_attack + 1e-9), 0.1, 10.0))]
    print(f"[Runner] Class weights: legit={class_weights[0]:.2f}, attack={class_weights[1]:.2f}")

    # ── Vehicle setup ─────────────────────────────────────────────────────
    all_vids    = list(vehicle_data_train.keys())[:N_VEHICLES]
    n_total     = len(all_vids)
    n_attackers = int(n_total * ATTACKER_FRACTION)
    attacker_vids = set(all_vids[-n_attackers:])

    def assign_role(vid):
        h = hash(str(vid)) % 10
        if h < 2:  return "infrastructure"
        if h < 5:  return "fleet"
        return "private"

    print(f"[Runner] {n_total} vehicles, {n_attackers} attackers ({ATTACKER_FRACTION*100:.0f}%)")

    # ── Initialise your model ─────────────────────────────────────────────
    rsu   = RSUDT(eps_min=EPS_MIN, eps_max=EPS_MAX, clip_C=CLIP_C,
                  min_history=MIN_TWIN_HISTORY)
    cloud = CloudDT()

    vehicles = {}
    for vid in all_vids:
        role        = assign_role(vid)
        is_attacker = vid in attacker_vids
        attack_type = vehicle_data_train[vid]["attack_type"] if is_attacker else 0
        v = Vehicle(vid, role=role, eps_total=EPS_TOTAL,
                    is_attacker=is_attacker, attack_type=int(attack_type))
        if is_attacker:
            v.trust_score = 0.30
        vehicles[vid] = v
        rsu.register_vehicle(vid, role, v.trust_score, eps_total=EPS_TOTAL)

    # ── Initialise baselines ──────────────────────────────────────────────
    bl_fldp     = FLDPBaseline(fixed_eps=FIXED_EPS_BASELINE, clip_C=CLIP_C)
    bl_dtfl     = DTFLBaseline()
    bl_adaptive = AdaptiveDPFLBaseline(eps_min=EPS_MIN, eps_max=EPS_MAX, clip_C=CLIP_C)

    # ── Tracking lists — all initialised here ────────────────────────────
    round_stats         = []
    vehicle_anomaly_log = []   # ← ground-truth anomaly log for evaluate_anomaly.py

    print(f"\n[Runner] Starting {N_ROUNDS} FL rounds...\n")

    # ── FL rounds ─────────────────────────────────────────────────────────
    for rnd in tqdm(range(1, N_ROUNDS + 1), desc="FL Rounds"):

        n_active    = max(4, int(len(all_vids) * 0.80))
        active_vids = np.random.choice(all_vids, size=n_active, replace=False).tolist()

        your_updates         = []
        baseline_vehicle_data = []

        for vid in active_vids:
            vdata = vehicle_data_train.get(vid)
            if vdata is None:
                continue
            X_full, y_full = vdata["X"], vdata["y"]
            if len(X_full) < 4:
                continue

            # Sample beacons
            n_sample = min(BEACONS_PER_ROUND, len(X_full))
            idx = np.random.choice(len(X_full), size=n_sample, replace=True)
            X_s, y_s = X_full[idx], y_full[idx]

            # single-class samples allowed

            # ── Your model ────────────────────────────────────────────────
            v        = vehicles[vid]
            twin     = rsu.twins[vid]
            # Reset budget at start of each round (Dwork & Roth 2014)
            rsu.reset_round_budget(twin)
            envelope = cloud.get_policy_for_role(v.role)
            eps_v, _ = rsu.schedule_eps(twin, eps_envelope=envelope)

            if v.rbac_eps_check(eps_v, CLIP_C):
                update = v.local_train(X_s, y_s, cloud.global_flat_params,
                                       epochs=LOCAL_EPOCHS, lr=LR,
                                       class_weights=class_weights)
                flags = rsu.screen_gradient(
                    twin, update["gradient"], update["tier_tag"], update["eps_consumed"])
                rsu.update_trust(twin, flags)
                update["flags"] = flags

                # ── Ground-truth anomaly log ──────────────────────────────
                # attack_type comes from the VeReMi 'class' column.
                # flagged = anomaly detector fired this round.
                # This is used by evaluate_anomaly.py to compute precision/recall
                # against VeReMi ground truth.
                vehicle_anomaly_log.append({
                    "vid":         str(vid),
                    "round":       rnd,
                    "flagged":     len(flags) > 0,       # what detector said
                    "attack_type": int(vdata["attack_type"]),  # VeReMi ground truth
                    "flags":       flags,
                })

                your_updates.append(update)

            baseline_vehicle_data.append({"X": X_s, "y": y_s, "vid": vid})

        # ── RSU aggregation + cloud update ────────────────────────────────
        if your_updates:
            agg_grad   = rsu.aggregate(your_updates)
            zone_meta  = rsu.get_zone_metadata(your_updates)
            zone_meta["gradient"] = agg_grad

            cloud.aggregate_zones([zone_meta])
            current_loss = cloud.compute_loss(X_test_all, y_test_all)
            stagnating   = cloud.convergence_check(current_loss)
            cloud.update_eps_envelope(zone_meta["anomaly_rate"], stagnating)

            new_thr = cloud.attack_simulation_update(zone_meta["anomaly_rate"])
            rsu.norm_sigma_mult = new_thr["norm_sigma_mult"]
            rsu.cosine_floor    = new_thr["cosine_floor"]

        # ── Baselines ─────────────────────────────────────────────────────
        bl_fldp.run_round(baseline_vehicle_data,     class_weights=class_weights)
        bl_dtfl.run_round(baseline_vehicle_data,     class_weights=class_weights)
        bl_adaptive.run_round(baseline_vehicle_data, class_weights=class_weights)

        # ── Per-round metrics ─────────────────────────────────────────────
        m_yours    = evaluate(lambda x: predict_with_model(cloud.global_model, x),
                              X_test_all, y_test_all)
        m_fldp     = evaluate(bl_fldp,     X_test_all, y_test_all)
        m_dtfl     = evaluate(bl_dtfl,     X_test_all, y_test_all)
        m_adaptive = evaluate(bl_adaptive, X_test_all, y_test_all)

        flagged   = sum(1 for u in your_updates if u.get("flags")) if your_updates else 0
        avg_trust = float(np.mean([rsu.twins[v].trust_score
                                   for v in active_vids if v in rsu.twins]))

        round_stats.append({
            "round":             rnd,
            "your_accuracy":     m_yours["accuracy"],
            "your_f1":           m_yours["f1"],
            "fl_dp_accuracy":    m_fldp["accuracy"],
            "fl_dp_f1":          m_fldp["f1"],
            "dt_fl_accuracy":    m_dtfl["accuracy"],
            "dt_fl_f1":          m_dtfl["f1"],
            "adaptive_accuracy": m_adaptive["accuracy"],
            "adaptive_f1":       m_adaptive["f1"],
            "flagged":           flagged,
            "avg_trust":         avg_trust,
            "n_active":          n_active,
        })

        if rnd % 5 == 0 or rnd == 1:
            tqdm.write(
                f"  Round {rnd:2d} | "
                f"Yours: acc={m_yours['accuracy']:.3f} f1={m_yours['f1']:.3f} | "
                f"FL+DP: acc={m_fldp['accuracy']:.3f} | "
                f"DT+FL: acc={m_dtfl['accuracy']:.3f} | "
                f"Adaptive: acc={m_adaptive['accuracy']:.3f} | "
                f"Flagged: {flagged}/{len(your_updates) if your_updates else 0} | "
                f"AvgTrust: {avg_trust:.3f}"
            )

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION (full test set)")
    print("=" * 65)

    final_results = {}
    model_map = {
        "DT-RBAC-FL-ADP (yours)": lambda x: predict_with_model(cloud.global_model, x),
        "FL + DP":                 bl_fldp.predict,
        "DT + FL":                 bl_dtfl.predict,
        "Adaptive DP-FL":          bl_adaptive.predict,
    }

    for name, pred_fn in model_map.items():
        preds = pred_fn(X_test_all)
        acc   = accuracy_score(y_test_all, preds)
        f1    = f1_score(y_test_all, preds, zero_division=0)
        prec  = precision_score(y_test_all, preds, zero_division=0)
        rec   = recall_score(y_test_all, preds, zero_division=0)
        final_results[name] = {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

        print(f"\n{name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(classification_report(y_test_all, preds,
                                    target_names=["Legitimate", "Attack"],
                                    zero_division=0))

    # ── Save all results ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "round_stats.json"), "w") as f:
        json.dump(round_stats, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    # vehicle_anomaly_log: every round × vehicle entry with VeReMi ground truth
    # used by evaluate_anomaly.py to compute anomaly detection precision/recall
    with open(os.path.join(RESULTS_DIR, "vehicle_anomaly_log.json"), "w") as f:
        json.dump(vehicle_anomaly_log, f, indent=2)

    trust_evolution = {vid: rsu.twins[vid].trust_score
                       for vid in list(rsu.twins.keys())[:20]}
    with open(os.path.join(RESULTS_DIR, "trust_scores.json"), "w") as f:
        json.dump(trust_evolution, f, indent=2)

    print(f"\n[Runner] Results saved to {RESULTS_DIR}/")
    print("  - round_stats.json")
    print("  - final_results.json")
    print("  - vehicle_anomaly_log.json  ← for evaluate_anomaly.py")
    print("  - trust_scores.json")
    print("\nNext steps:")
    print("  python evaluate_anomaly.py   ← anomaly detection vs VeReMi labels")
    print("  python plot_results.py       ← charts\n")

    return round_stats, final_results


if __name__ == "__main__":
    main()
