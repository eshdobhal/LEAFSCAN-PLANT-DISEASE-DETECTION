"""Write demo/field_history.csv: 20 days of one SIMULATED field (surveillance/progression.py) for the forecast demo.

The example is chosen to show the alert: on its last observed day the field is below the 5% action threshold,
it truly crosses it within the next 7 days, and the trained LSTM flags it (a true positive). The LSTM misses
about a third of such fields (alert recall 0.68, docs/TECHNICAL_REPORT.md), so this is a best case, not typical.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surveillance.progression import LOOKBACK, HORIZON, THRESHOLD, Forecaster, simulate  # noqa: E402

seq, truth = simulate(100, seed=7, noise=0.0485)
net = Forecaster("lstm")
net.load_state_dict(torch.load(ROOT / "surveillance" / "runs" / "progression.pt", map_location="cpu", weights_only=True)["lstm"])
net.eval()
found = None
with torch.no_grad():
    for i in range(len(seq)):
        for end in range(30, 80):  # last observed day (exclusive)
            x = seq[i, end - LOOKBACK:end]
            if x[-1, 0] < 0.03 and truth[i, end:end + HORIZON].max() >= THRESHOLD \
                    and float(net(torch.from_numpy(x)[None]).max()) >= THRESHOLD:
                found = (i, end)
                break
        if found:
            break
i, end = found
rows = seq[i, end - 20:end]
lines = ["day,severity,temperature_c,humidity_pct,rain_mm,spray"]
for d, (sev, t, h, r, s) in enumerate(rows, 1):
    lines.append(f"{d},{sev:.4f},{t * 6 + 18:.1f},{h * 15 + 70:.0f},{r * 10:.1f},{int(s)}")
(ROOT / "demo" / "field_history.csv").write_text("\n".join(lines) + "\n")
print(f"field {i}, days {end - 20}-{end - 1}; true severity next 7 days (%):",
      [round(float(v) * 100, 2) for v in truth[i, end:end + HORIZON]])
