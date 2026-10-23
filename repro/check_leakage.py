"""B5/E1 leakage check (CPU, 1 s).  python repro/check_leakage.py
Reports whether any file fitted OUTSIDE the nested folds can reach the reported numbers:
 - models/rule_weights.json : build_features / robustness / graph_diag load it through hlmrf.learned_weights. If it exists and differs from
   hlmrf.init_weights, the 'hand weight' rows are NOT hand weights (they were learned on the training split).
 - models/calibrators.pkl(+_lr.pkl), models/crc.json : legacy dev-fitted artefacts; not used by build_features (Calibrators('none')) or by nested CV."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from cprove.utils import load_config
cfg = load_config("configs/frozen.yaml"); h = cfg["hlmrf"]; init = h["init_weights"]
p = Path(h.get("learned_weights") or "models/rule_weights.json")
print(f"{p}: {'EXISTS' if p.exists() else 'absent'}")
leak = False
if p.exists():
    w = json.load(open(p)); diff = {k: (init.get(k), round(v, 3)) for k, v in w.items() if abs(init.get(k, v) - v) > 1e-3}
    leak = bool(diff); print("  differs from init_weights:" if diff else "  identical to init_weights (no effect)", diff or "")
for f in ("models/calibrators.pkl", "models/calibrators.pkl_lr.pkl", "models/crc.json"):
    print(f"{f}: {'EXISTS (legacy, unused by the reported tables; remove from the release)' if Path(f).exists() else 'absent'}")
print("\nRESULT:", "LEAK - hand-weight rows used learned weights. Run: python -m train.recompute_hand, then rerun nested_cv/full_metrics/graph_eval/ladder." if leak else "OK - hand-weight rows use the configured init weights.")
