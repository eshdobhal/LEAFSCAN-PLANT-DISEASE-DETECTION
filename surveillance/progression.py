"""Disease-progression forecasting with an RNN and an LSTM (brief item 9).

    python -m surveillance.progression

IMPORTANT: no public dataset follows the same diseased plants photo by photo over a season, so the
sequences here are SIMULATED from an established epidemiological model, not observed. What is real: the
model structure (Vanderplank's logistic disease-progress curve), the weather-dependence of late blight
(fastest at ~15-25 C and high humidity), and observation noise matching LeafScan's measured severity error.

Simulation per field (90 days): daily temperature, humidity and rain -> infection rate r_t; severity
y_{t+1} = y_t + r_t * y_t * (1 - y_t); a fungicide spray cuts r_t by 80% for 7 days; observations add noise.
Task: from the last 14 days (observed severity, weather, spray flag) predict severity for each of the next
7 days, and whether it will pass the 5% action threshold within 7 days.
Models: persistence, logistic-curve extrapolation, vanilla RNN (tanh), LSTM. Splits are by field.
Writes surveillance/results/progression_*.png and progression.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "surveillance" / "results"
DAYS, LOOKBACK, HORIZON, THRESHOLD = 90, 14, 7, 0.05
N_FIELDS = {"train": 800, "val": 100, "test": 200}
FEATURES = ["observed severity", "temperature", "humidity", "rain", "spray"]


def observation_noise():
    """Severity measurement error of the segmentation model, if it has been evaluated; else 3 points."""
    f = ROOT / "surveillance" / "runs" / "leafseg_deeplab" / "test_metrics.json"
    return float(json.loads(f.read_text())["severity_fraction_MAE"]) if f.exists() else 0.03


def simulate(n, seed, noise):
    rng = np.random.default_rng(seed)
    t = np.arange(DAYS)
    seq = np.zeros((n, DAYS, len(FEATURES)), np.float32)
    truth = np.zeros((n, DAYS), np.float32)
    for i in range(n):
        temp = 18 + 6 * np.sin(2 * np.pi * (t / DAYS + rng.uniform())) + rng.normal(0, 2.5, DAYS)
        humid = np.clip(70 + 15 * np.sin(2 * np.pi * (t / 30 + rng.uniform())) + rng.normal(0, 8, DAYS), 30, 100)
        rain = (rng.random(DAYS) < 0.25) * rng.gamma(2, 4, DAYS)
        spray = np.zeros(DAYS)
        for day in rng.choice(DAYS, rng.integers(0, 4), replace=False):
            spray[day] = 1
        # Weather suitability for late blight: temperature bell around 20 C, humidity above 75 %, rain helps.
        temp_f = np.exp(-((temp - 20) / 5) ** 2)
        humid_f = 1 / (1 + np.exp(-(humid - 78) / 4))
        r = 0.32 * temp_f * (0.3 + 0.7 * humid_f) * (1 + 0.3 * (rain > 2))
        protect = np.convolve(spray, np.ones(7), mode="full")[:DAYS] > 0  # 7 days of fungicide protection
        r = np.where(protect, r * 0.2, r)
        y = np.zeros(DAYS)
        y[0] = rng.uniform(0.0005, 0.005)  # first lesions
        for d in range(DAYS - 1):
            y[d + 1] = min(1.0, y[d] + r[d] * y[d] * (1 - y[d]))
        obs = np.clip(y + rng.normal(0, noise * 0.5, DAYS) + noise * 0.3 * rng.standard_t(3, DAYS) * (y > 0.01), 0, 1)
        seq[i] = np.stack([obs, (temp - 18) / 6, (humid - 70) / 15, rain / 10, spray], 1)
        truth[i] = y
    return seq, truth


def windows(seq, truth):
    xs, ys = [], []
    for start in range(0, DAYS - LOOKBACK - HORIZON + 1, 3):
        xs.append(seq[:, start:start + LOOKBACK])
        ys.append(truth[:, start + LOOKBACK:start + LOOKBACK + HORIZON])
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))


class Forecaster(nn.Module):
    def __init__(self, cell="lstm", hidden=64):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn(len(FEATURES), hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = x[:, -1, 0:1]
        return (last + self.head(out[:, -1])).clamp(0, 1)  # predict the change from today's observed severity


def fit(cell, train, val, epochs=40, seed=0):
    torch.manual_seed(seed)
    model = Forecaster(cell)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train), batch_size=256, shuffle=True)
    best, best_state, history = 1e9, None, []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.functional.l1_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mae = float(nn.functional.l1_loss(model(val[0]), val[1]))
        history.append(round(val_mae, 5))
        if val_mae < best:
            best, best_state = val_mae, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), history


def logistic_extrapolation(x):
    """Fit r from the last 7 observed days of the logistic curve (in logit space) and project forward."""
    obs = x[:, -7:, 0].numpy().clip(1e-4, 1 - 1e-4)
    logit = np.log(obs / (1 - obs))
    slope = np.polyfit(np.arange(7), logit.T, 1)[0].clip(0, None)  # disease does not heal: r >= 0
    last = logit[:, -1]
    future = last[:, None] + slope[:, None] * np.arange(1, HORIZON + 1)[None]
    return torch.from_numpy(1 / (1 + np.exp(-future))).float()


def evaluate(pred, true, x):
    mae_h = (pred - true).abs().mean(0).numpy()
    now_below = x[:, -1, 0] < THRESHOLD
    will = (true.max(1).values >= THRESHOLD) & now_below
    said = (pred.max(1).values >= THRESHOLD) & now_below
    tp, fp, fn = int((will & said).sum()), int((~will & said).sum()), int((will & ~said).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"MAE_all_horizons": round(float(mae_h.mean()), 4), "MAE_day1": round(float(mae_h[0]), 4),
            "MAE_day7": round(float(mae_h[-1]), 4), "MAE_by_day": [round(float(v), 4) for v in mae_h],
            "threshold_alert_precision": round(prec, 4), "threshold_alert_recall": round(rec, 4),
            "threshold_alert_F1": round(2 * prec * rec / max(1e-9, prec + rec), 4)}


def gate_activations(model, x):
    """Run one sequence step by step through the first LSTM layer and record its gates."""
    lstm = model.rnn
    w_ih, w_hh, b = lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0 + lstm.bias_hh_l0
    h = torch.zeros(lstm.hidden_size)
    c = torch.zeros(lstm.hidden_size)
    gates = {"input (sigmoid)": [], "forget (sigmoid)": [], "candidate (tanh)": [], "output (sigmoid)": []}
    with torch.no_grad():
        for step in x:
            z = w_ih @ step + w_hh @ h + b
            i, f, g, o = z.chunk(4)
            i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            for k, v in zip(gates, (i, f, g, o)):
                gates[k].append(v.numpy())
    return {k: np.stack(v) for k, v in gates.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RESULTS.mkdir(parents=True, exist_ok=True)
    noise = observation_noise()
    data = {s: simulate(n, seed, noise) for (s, n), seed in zip(N_FIELDS.items(), (1, 2, 3))}
    win = {s: windows(*data[s]) for s in data}
    out = {"simulated": True, "observation_noise_severity_MAE": noise, "fields": N_FIELDS,
           "lookback_days": LOOKBACK, "horizon_days": HORIZON, "alert_threshold": THRESHOLD,
           "test_windows": int(len(win["test"][0]))}
    xt, yt = win["test"]
    results = {"persistence (tomorrow = today)": evaluate(xt[:, -1:, 0].repeat(1, HORIZON).clamp(0, 1), yt, xt),
               "logistic-curve extrapolation": evaluate(logistic_extrapolation(xt), yt, xt)}
    models, histories = {}, {}
    for cell, label in (("rnn", "vanilla RNN (tanh)"), ("lstm", "LSTM")):
        m, hist = fit(cell, win["train"], win["val"])
        models[cell], histories[label] = m, hist
        with torch.no_grad():
            results[label] = evaluate(m(xt), yt, xt)
    out["results"] = results
    for k, v in results.items():
        print(f"{k:32s} MAE {v['MAE_all_horizons']:.4f} (day1 {v['MAE_day1']:.4f}, day7 {v['MAE_day7']:.4f})  alert F1 {v['threshold_alert_F1']:.3f}")

    colours = ["#898781", "#eda100", "#eb6834", "#2a78d6"]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for (k, v), col in zip(results.items(), colours):
        ax.plot(range(1, HORIZON + 1), np.array(v["MAE_by_day"]) * 100, "o-", color=col, lw=2, label=k)
    ax.set_xlabel("days ahead")
    ax.set_ylabel("mean absolute error (% of leaf)")
    ax.set_title("Forecast error by horizon (simulated fields)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "progression_error_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example fields: full truth, observations, and the LSTM's 7-day forecast from day 40.
    seq, truth = data["test"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, i in zip(axes, (0, 7, 21)):
        ax.plot(truth[i] * 100, color="#0b0b0b", lw=1.5, label="true severity")
        ax.plot(seq[i, :, 0] * 100, ".", color="#898781", ms=3, label="observed (noisy)")
        for day in np.where(seq[i, :, 4] > 0)[0]:
            ax.axvline(day, color="#1baf7a", lw=1, alpha=0.7)
        start = 40
        with torch.no_grad():
            f_lstm = models["lstm"](torch.from_numpy(seq[i:i + 1, start - LOOKBACK:start]))[0].numpy()
            f_rnn = models["rnn"](torch.from_numpy(seq[i:i + 1, start - LOOKBACK:start]))[0].numpy()
        days = np.arange(start, start + HORIZON)
        ax.plot(days, f_lstm * 100, color="#2a78d6", lw=2.5, label="LSTM forecast")
        ax.plot(days, f_rnn * 100, color="#eb6834", lw=2, ls="--", label="RNN forecast")
        ax.axhline(THRESHOLD * 100, color="#d03b3b", lw=0.8, ls=":")
        ax.set_xlabel("day")
        ax.set_title(f"simulated field {i}", loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("severity (% of leaf damaged)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Disease progression: green lines = fungicide sprays, red dotted = 5% action threshold (SIMULATED data)",
                 x=0.01, ha="left", fontsize=9)
    fig.savefig(RESULTS / "progression_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # LSTM gates over one sequence: sigmoid gates live in (0, 1), the tanh candidate in (-1, 1).
    g = gate_activations(models["lstm"], torch.from_numpy(seq[7, 26:26 + LOOKBACK]))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (k, v) in zip(axes, g.items()):
        im = ax.imshow(v[:, :24].T, aspect="auto", cmap="RdBu_r" if "tanh" in k else "viridis",
                       vmin=-1 if "tanh" in k else 0, vmax=1)
        ax.set_title(k, loc="left", fontsize=9)
        ax.set_xlabel("day in the 14-day window")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("hidden unit (first 24)")
    fig.suptitle("Inside the LSTM: gate activations over a sequence", x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / "progression_lstm_gates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    out["gate_mean_activation"] = {k: round(float(v.mean()), 3) for k, v in g.items()}
    out["validation_MAE_curves"] = histories
    torch.save({k: m.state_dict() for k, m in models.items()}, ROOT / "surveillance" / "runs" / "progression.pt")
    (RESULTS / "progression.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
