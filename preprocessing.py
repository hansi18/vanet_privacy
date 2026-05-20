"""
preprocessing.py
Loads VeReMi Extension dataset (CSV or JSON) and engineers 4 features.
Handles the exact VeReMi Extension column layout:
  type, sendTime, sender, senderPseudo, messageID, class,
  posx, posy, spdx, spdy, hedx, hedy, ...
"""

import os, json, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["pos_zscore", "speed_anomaly", "heading_dev", "time_delta"]
LABEL_MAP    = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(path):
    if path.lower().endswith(".csv") and os.path.isfile(path):
        return "csv"
    csvs  = glob.glob(os.path.join(path,"**","*.csv"),recursive=True) + \
            glob.glob(os.path.join(path,"*.csv"))
    if csvs: return "csv"
    jsons = glob.glob(os.path.join(path,"**","*.json"),recursive=True) + \
            glob.glob(os.path.join(path,"*.json"))
    if jsons: return "json"
    raise FileNotFoundError(
        f"No CSV or JSON files found at: {path}\n"
        "Check DATA_PATH in config.py.")


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(data_path, max_vehicles, verbose):
    if data_path.lower().endswith(".csv") and os.path.isfile(data_path):
        csv_files = [data_path]
    else:
        csv_files = sorted(set(
            glob.glob(os.path.join(data_path,"**","*.csv"),recursive=True) +
            glob.glob(os.path.join(data_path,"*.csv"))))

    if verbose:
        print(f"[Preprocessing] Found {len(csv_files)} CSV file(s)")

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"[Preprocessing] Warning: skipping {f}: {e}")
    if not dfs:
        raise ValueError("CSV files found but none could be read.")

    df = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"[Preprocessing] Raw columns: {list(df.columns)}")
        print(f"[Preprocessing] Raw shape:   {df.shape}")

    # ── Lowercase all column names for matching ───────────────────────────
    df.columns = [c.strip().lower() for c in df.columns]
    cols = set(df.columns)

    # ── vehicle_id ────────────────────────────────────────────────────────
    # VeReMi Extension: 'sender' is the numeric vehicle ID
    # 'senderpseudo' is the pseudonym — also useful
    vid_col = None
    for cand in ["sender", "senderid", "senderpseudo", "vehicle_id",
                 "id", "node_id", "vid"]:
        if cand in cols:
            vid_col = cand
            break
    if vid_col:
        df["vehicle_id"] = df[vid_col].astype(str)
    else:
        df["vehicle_id"] = "v0"

    # ── attack_type ───────────────────────────────────────────────────────
    # VeReMi Extension: 'class' is the attack label (0=legit, 1-5=attack)
    # 'type' is BSM message type — do NOT use it as attack label
    atk_col = None
    for cand in ["class", "attack_type", "label", "attacktype",
                 "misbehaviortype", "attacklabel"]:
        if cand in cols:
            atk_col = cand
            break
    df["attack_type"] = pd.to_numeric(
        df[atk_col] if atk_col else 0, errors="coerce").fillna(0).astype(int)

    # ── timestamp ─────────────────────────────────────────────────────────
    ts_col = None
    for cand in ["sendtime", "rcvtime", "timestamp", "time", "simtime"]:
        if cand in cols:
            ts_col = cand
            break
    df["timestamp"] = pd.to_numeric(
        df[ts_col] if ts_col else pd.Series(np.arange(len(df))),
        errors="coerce").fillna(0)

    # ── position x ───────────────────────────────────────────────────────
    for cand in ["posx", "pos_x", "x", "position_x", "sendpos_x"]:
        if cand in cols:
            df["x"] = pd.to_numeric(df[cand], errors="coerce").fillna(0)
            break
    if "x" not in df.columns:
        df["x"] = 0.0

    # ── position y ───────────────────────────────────────────────────────
    for cand in ["posy", "pos_y", "y", "position_y", "sendpos_y"]:
        if cand in cols:
            df["y"] = pd.to_numeric(df[cand], errors="coerce").fillna(0)
            break
    if "y" not in df.columns:
        df["y"] = 0.0

    # ── speed — compute from spdx/spdy components ─────────────────────────
    if "spdx" in cols and "spdy" in cols:
        df["speed"] = np.sqrt(
            pd.to_numeric(df["spdx"], errors="coerce").fillna(0)**2 +
            pd.to_numeric(df["spdy"], errors="coerce").fillna(0)**2)
    else:
        spd_col = next((c for c in ["spd","speed","vel","velocity"] if c in cols), None)
        df["speed"] = pd.to_numeric(df[spd_col] if spd_col else 0,
                                    errors="coerce").fillna(0).abs()

    # ── heading — compute from hedx/hedy components ───────────────────────
    if "hedx" in cols and "hedy" in cols:
        df["heading"] = np.degrees(np.arctan2(
            pd.to_numeric(df["hedy"], errors="coerce").fillna(0),
            pd.to_numeric(df["hedx"], errors="coerce").fillna(0)))
    else:
        hed_col = next((c for c in ["heading","angle","dir","yaw"] if c in cols), None)
        df["heading"] = pd.to_numeric(df[hed_col] if hed_col else 0,
                                      errors="coerce").fillna(0)

    # ── Ensure numeric & drop bad rows ───────────────────────────────────
    for col in ["x","y","speed","heading","timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
    df = df[df["vehicle_id"] != ""].copy()

    # ── Verify we have multiple vehicles ─────────────────────────────────
    all_vids = df["vehicle_id"].unique()
    if verbose:
        print(f"[Preprocessing] Unique vehicle IDs: {len(all_vids)}")

    if len(all_vids) <= 2 and "senderpseudo" in cols:
        # Fall back to pseudonym
        df["vehicle_id"] = df["senderpseudo"].astype(str)
        all_vids = df["vehicle_id"].unique()
        if verbose:
            print(f"[Preprocessing] Using 'senderpseudo' → {len(all_vids)} vehicles")

    if len(all_vids) <= 2:
        # Last resort: row-chunking
        chunk = 500
        if verbose:
            print(f"[Preprocessing] Row-chunking fallback ({chunk} rows/vehicle)")
        df["vehicle_id"] = (np.arange(len(df)) // chunk).astype(str)
        all_vids = df["vehicle_id"].unique()

    # ── Select up to max_vehicles, cap rows per vehicle ───────────────────
    if len(all_vids) > max_vehicles:
        df = df[df["vehicle_id"].isin(all_vids[:max_vehicles])].copy()

    # Cap rows per vehicle (pandas groupby.apply drops group col in newer versions)
    parts = []
    for vid, grp in df.groupby("vehicle_id"):
        parts.append(grp.head(2000))
    df = pd.concat(parts).reset_index(drop=True)

    if verbose:
        print(f"[Preprocessing] Final: {len(df):,} rows, "
              f"{df['vehicle_id'].nunique()} vehicles")
        vc = df.drop_duplicates("vehicle_id").groupby("attack_type")["vehicle_id"].count()
        print(f"[Preprocessing] Attack type distribution: {vc.to_dict()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# JSON loading (unchanged, kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_file(filepath):
    records = []
    try:
        with open(filepath,"r") as f:
            data = json.loads(f.read().strip())
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict): continue
            vid   = str(item.get("id", item.get("senderId","unknown")))
            atype = int(item.get("type", item.get("attackType", 0)))
            for msg in item.get("messages", item.get("traces",[])):
                if not isinstance(msg, dict): continue
                pos = msg.get("pos",[0,0,0])
                if isinstance(pos, dict):
                    pos = [pos.get("x",0), pos.get("y",0), 0]
                spd = msg.get("spd", msg.get("speed",0))
                if isinstance(spd, list):
                    spd = float(np.linalg.norm(spd[:2]))
                records.append({
                    "vehicle_id": vid, "attack_type": atype,
                    "timestamp": float(msg.get("rcvTime", msg.get("time",0))),
                    "x": float(pos[0]) if len(pos)>0 else 0.0,
                    "y": float(pos[1]) if len(pos)>1 else 0.0,
                    "speed": float(spd),
                    "heading": float(msg.get("heading",0)),
                })
    except Exception:
        pass
    return records


def _load_json(data_path, max_vehicles, verbose):
    files = sorted(set(
        glob.glob(os.path.join(data_path,"**","*.json"),recursive=True) +
        glob.glob(os.path.join(data_path,"*.json"))))
    if verbose:
        print(f"[Preprocessing] Found {len(files)} JSON file(s)")
    records, seen = [], set()
    for f in files:
        recs = _parse_json_file(f)
        for r in recs: seen.add(r["vehicle_id"])
        records.extend(recs)
        if len(seen) >= max_vehicles: break
    df = pd.DataFrame(records).dropna()
    df = df[df["vehicle_id"].isin(list(seen)[:max_vehicles])]
    if verbose:
        print(f"[Preprocessing] Loaded {len(df):,} beacons, "
              f"{df['vehicle_id'].nunique()} vehicles")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_veremi(data_path, max_vehicles=200, verbose=True):
    """Load VeReMi from a CSV file, CSV folder, or JSON folder."""
    fmt = _detect_format(data_path)
    if verbose:
        print(f"[Preprocessing] Detected format: {fmt.upper()}")
    df = _load_csv(data_path, max_vehicles, verbose) if fmt=="csv" \
         else _load_json(data_path, max_vehicles, verbose)
    for col in ["vehicle_id","attack_type","timestamp","x","y","speed","heading"]:
        if col not in df.columns:
            df[col] = 0
    return df.reset_index(drop=True)


def engineer_features(df, verbose=True):
    if verbose:
        print("[Preprocessing] Engineering features...")
    parts = []
    for vid, grp in df.sort_values("timestamp").groupby("vehicle_id"):
        grp = grp.reset_index(drop=True).copy()
        dists = np.sqrt(grp["x"].diff()**2 + grp["y"].diff()**2).fillna(0)
        mu, sig = dists.mean(), dists.std()+1e-9
        grp["pos_zscore"]    = (dists - mu) / sig
        rm = grp["speed"].rolling(5,min_periods=1).mean()
        rs = grp["speed"].rolling(5,min_periods=1).std().fillna(1e-9)+1e-9
        grp["speed_anomaly"] = ((grp["speed"]-rm)/rs).abs()
        grp["heading_dev"]   = grp["heading"].diff().abs().fillna(0)
        grp["time_delta"]    = grp["timestamp"].diff().fillna(0).abs()
        parts.append(grp)

    result = pd.concat(parts).reset_index(drop=True)
    result["label"] = result["attack_type"].map(lambda x: LABEL_MAP.get(int(x),1))
    for col in FEATURE_COLS:
        lo, hi = result[col].quantile(0.01), result[col].quantile(0.99)
        result[col] = result[col].clip(lo, hi)

    if verbose:
        legit  = (result["label"]==0).sum()
        attack = (result["label"]==1).sum()
        print(f"[Preprocessing] Features ready — legit:{legit:,}  "
              f"attack:{attack:,}  ratio:{attack/(legit+attack)*100:.1f}%")
    return result


def prepare_vehicle_datasets(df, scaler=None, fit_scaler=True):
    df = df.copy()
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    else:
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    vdata = {}
    for vid, grp in df.groupby("vehicle_id"):
        vdata[vid] = {
            "X": grp[FEATURE_COLS].values.astype(np.float32),
            "y": grp["label"].values.astype(np.int64),
            "attack_type": int(grp["attack_type"].iloc[0]),
        }
    return vdata, scaler
