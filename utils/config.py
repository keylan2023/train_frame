import argparse
import yaml
import copy
import json

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _set_by_path(d, key_path, value):
    """Set nested dict/list value by dot path (creates intermediate containers)."""
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        # list index
        if isinstance(cur, list) and p.isdigit():
            idx = int(p)
            while len(cur) <= idx:
                cur.append({})
            cur = cur[idx]
            continue
        # dict key
        if p not in cur or not isinstance(cur[p], (dict, list)):
            cur[p] = {}
        cur = cur[p]
    last = parts[-1]

    # Detect desired type from existing value if present
    def _cast(v, ref):
        if ref is None: 
            # try JSON for lists/dicts/bools/numbers
            try:
                return json.loads(v)
            except Exception:
                return v
        if isinstance(ref, bool):
            if isinstance(v, str):
                return v.lower() in ["1","true","yes","y","on"]
            return bool(v)
        if isinstance(ref, int):
            try: return int(v)
            except: return v
        if isinstance(ref, float):
            try: return float(v)
            except: return v
        if isinstance(ref, list):
            try:
                out = json.loads(v)
                if isinstance(out, list): return out
            except Exception:
                pass
            return [v]
        return v

    if isinstance(cur, list) and last.isdigit():
        idx = int(last)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    else:
        ref = None
        if isinstance(cur, dict):
            ref = cur.get(last, None)
        cur[last] = _cast(value, ref)

def parse_args_and_config():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("overrides", nargs="*", help="Dotlist overrides like a.b=2 c.d=true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    cfg = copy.deepcopy(cfg)
    for item in args.overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        _set_by_path(cfg, k, v)
    return cfg
