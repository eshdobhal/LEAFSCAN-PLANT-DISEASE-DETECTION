"""Assemble the demo video docs/demo/leafscan_demo.mp4 (1280x720, H.264) from material made by the other
scripts in demo/ (run them first, see demo/README.md):

  1. the live browser version (record_browser.py: real screen recording of the GitHub Pages site)
  2. the surveillance CLI on photos, the tea pest mode and the 7-day forecast (real CLI output and reports)
  3. the video mode on demo/field_walk_SYNTHETIC.mp4 (a SYNTHETIC clip assembled from still photos)
  4. headline results, read from the result files (nothing typed in by hand), and the limits

    .venv-train/Scripts/python demo/make_video.py
"""
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
OUT = ROOT / "docs" / "demo" / "leafscan_demo.mp4"
W, H, FPS = 1280, 720, 25
INK, MUTED, BG, ACCENT, PAPER = (11, 11, 11), (90, 88, 84), (249, 248, 245), (31, 122, 76), (255, 255, 255)


def font(size, mono=False, bold=False):
    names = ["consola.ttf"] if mono else (["segoeuib.ttf"] if bold else ["segoeui.ttf"])
    for n in names + ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class Video:
    """Pipes RGB frames to ffmpeg (bundled by the imageio-ffmpeg package) for a small H.264 MP4."""

    def __init__(self, path):
        import subprocess

        import imageio_ffmpeg
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                      "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                      str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def add(self, img, seconds=None, frames=1):
        raw = np.ascontiguousarray(np.asarray(img.convert("RGB").resize((W, H)))).tobytes()
        for _ in range(int(seconds * FPS) if seconds else frames):
            self.proc.stdin.write(raw)
            self.frames += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def card(title, lines=(), kicker=None, size=28, top=200, wrap=70):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = top
    if kicker:
        d.text((110, y - 50), kicker.upper(), fill=ACCENT, font=font(22, bold=True))
    d.text((110, y), title, fill=INK, font=font(54, bold=True))
    y += 100
    for line in lines:
        for part in textwrap.wrap(line, wrap, subsequent_indent="   ") or [""]:
            d.text((110, y), part, fill=MUTED, font=font(size))
            y += int(size * 1.5)
    return img


def caption(img, text):
    """Bottom caption bar over a frame."""
    img = img.convert("RGB").resize((W, H))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 64, W, H], fill=(11, 11, 11, 200))
    d.text((24, H - 50), text, fill=PAPER, font=font(26))
    return img


def fit(img, box_w, box_h):
    img = img.copy()
    img.thumbnail((box_w, box_h), Image.LANCZOS)
    return img


def terminal(command, output):
    img = Image.new("RGB", (W, H), (24, 24, 27))
    d = ImageDraw.Draw(img)
    f = font(21, mono=True)
    y = 40
    d.text((40, y), "PS> " + command, fill=(120, 200, 140), font=f)
    y += 44
    for line in output.splitlines():
        for part in textwrap.wrap(line, 100, subsequent_indent="    ") or [""]:
            d.text((40, y), part, fill=(255, 196, 120) if part.lstrip().startswith("!") else (230, 230, 230), font=f)
            y += 30
    return img


def showcase(path, text):
    canvas = Image.new("RGB", (W, H), BG)
    im = fit(Image.open(path).convert("RGB"), W - 80, H - 100)
    canvas.paste(im, ((W - im.width) // 2, 20))
    return caption(canvas, text)


def browser_section(v):
    webm = next((DEMO / "browser").glob("*.webm"))
    marks = json.loads((DEMO / "browser" / "marks.json").read_text())
    start = max(0.0, marks["model_ready"] - 2.0)
    cap = cv2.VideoCapture(str(webm))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        if t < start:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if t < marks["result"]:
            text = "Live site: upload a leaf photo. The model runs in the browser; the photo never leaves the phone."
        elif t < marks["result"] + 7:
            text = "Diagnosis, confidence, top-3 and what to do (a 'not sure' warning appears below 80%)"
        else:
            text = "Treatments page: every product, searchable and filtered by crop"
        v.add(caption(img, text))
    cap.release()


def forecast_chart(report_json, history_csv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = np.genfromtxt(history_csv, delimiter=",", names=True)
    f = json.loads(Path(report_json).read_text())["forecast"]
    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=100)
    days = rows["day"]
    ax.plot(days, rows["severity"] * 100, "o-", color="#2a78d6", lw=2, label="observed severity (simulated field)")
    fx = [days[-1]] + [days[-1] + d for d in f["days_ahead"]]
    fy = [rows["severity"][-1] * 100] + f["severity_percent"]
    ax.plot(fx, fy, "o--", color="#eb6834", lw=2, label="LSTM 7-day forecast")
    ax.axhline(5, color="#d03b3b", lw=1.2, ls=":")
    ax.text(days[0], 5.15, "5% action threshold", color="#d03b3b", fontsize=12)
    ax.set_xlabel("day", fontsize=13)
    ax.set_ylabel("leaf area damaged (%)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_title("Forecast: severity is predicted to cross the action threshold within 7 days", loc="left", fontsize=15)
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img


def results_lines():
    runs, sruns = ROOT / "training" / "runs", ROOT / "surveillance" / "runs"
    lines = []
    tm = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    ens = (runs / "ensemble" / "test_summary.md").read_text().splitlines()
    lab = next(line for line in ens if line.startswith("| test_lab_tta")).split("|")
    field = next(line for line in ens if line.startswith("| test_field_tta")).split("|")
    lines.append(f"Classification (39 classes): lab {lab[3].strip()} accuracy; real field photos {field[3].strip()}, macro F1 {field[6].strip()}")
    pd_ = tm(sruns / "plantdoc_retinanet" / "test_metrics.json")
    fr = tm(sruns / "plantdoc_fasterrcnn" / "test_metrics.json")
    if pd_:
        lines.append(f"Disease detection (PlantDoc, 29 classes): mAP@0.5 {pd_['mAP50']:.2f} RetinaNet"
                     + (f", {fr['mAP50']:.2f} Faster R-CNN" if fr else ""))
    tea = tm(sruns / "tea_retinanet" / "test_metrics.json")
    if tea:
        lines.append("Pest detection (tea field photos): AP@0.5 " + ", ".join(f"{k.split(' (')[0]} {v:.2f}" for k, v in tea["pest_AP50"].items()))
    seg = tm(sruns / "leafseg_deeplab" / "test_metrics.json")
    if seg:
        lines.append(f"Damaged tissue (DeepLabV3): IoU {seg['IoU_lesion']:.2f}; severity grade within ±1 for {seg['severity_grade_within_1']:.0%} of photos")
    prog = tm(ROOT / "surveillance" / "results" / "progression.json")
    if prog:
        r = prog["results"]
        lines.append(f"7-day forecast (simulated): LSTM MAE {r['LSTM']['MAE_all_horizons'] * 100:.1f} points vs persistence "
                     f"{r['persistence (tomorrow = today)']['MAE_all_horizons'] * 100:.1f}; alert F1 {r['LSTM']['threshold_alert_F1']:.2f}")
    ev = tm(ROOT / "surveillance" / "results" / "evaluation.json")
    if ev and "CAFormer ensemble (shipped)" in ev.get("classification", {}):
        f = ev["classification"]["CAFormer ensemble (shipped)"]["field"]
        lines.append(f"Calibration on field photos: ECE {f['ECE']:.3f} (target 0.10 or less)")
    if ev and "pipeline" in ev.get("pipeline_cpu_profile", {}):
        pp = ev["pipeline_cpu_profile"]["pipeline"]
        lines.append(f"Full pipeline on a 4-thread CPU: {pp['median_ms_per_component']['total (all five)'] / 1000:.2f} s per photo, "
                     f"{pp['peak_memory_MB'] / 1024:.2f} GB · 12 of 14 acceptance tests pass")
    return lines


def main():
    v = Video(OUT)
    v.add(card("LeafScan Surveillance", ["Crop disease and pest surveillance from phone photos and scouting video.",
                                         "Classify the disease · find diseased leaves and pests · measure damaged tissue · forecast the next 7 days.",
                                         "", "Esh Dobhal · Project 05, precision agriculture"], kicker="Demo"), 6)

    v.add(card("1. Live browser version", ["eshdobhal.github.io/LEAFSCAN-PLANT-DISEASE-DETECTION",
                                           "Nothing to install; works on a phone. (The one-time model download, about 60 s, is cut.)"],
               kicker="For growers"), 4)
    browser_section(v)

    v.add(card("2. Surveillance CLI", ["For scouts and agronomists: many photos at once, JSON reports, annotated images.",
                                       "Red = damaged tissue, blue boxes = diseased leaves, summary panel underneath."],
               kicker="For scouts"), 4)
    v.add(terminal("python -m surveillance.cli analyze demo\\photos\\*.jpg --out demo\\reports",
                   (DEMO / "cli_analyze.txt").read_text(encoding="utf-8")), 7)
    for name, text in (("leafnet75_00063", "Late blight: 34% of the leaf damaged, grade 4 of 5"),
                       ("leafnet75_00098", "Early blight: lesions outlined, severity 17%"),
                       ("leafnet75_00123", "Several leaves: each lesion is a separate region, severity graded 0-5"),
                       ("plantvillage_potato_late_blight", "A miss worth showing: right disease, but severity 0% (lab photo, unlike the field masks)")):
        v.add(showcase(DEMO / "reports" / f"{name}_annotated.png", text), 3.5)

    v.add(card("3. Pest detection", ["The pest model was trained on tea field photos (red spider mite, tea mosquito bug).",
                                     "On other crops it mistakes disease spots for pests (89% false alarms), so it runs only with --crop tea."],
               kicker="Tea only, on purpose"), 5)
    v.add(terminal("python -m surveillance.cli analyze tea_test_08048.jpg --crop tea", (DEMO / "cli_tea.txt").read_text(encoding="utf-8")), 4)
    v.add(showcase(DEMO / "reports" / "tea_test_08048_annotated.png", "Orange boxes: leaves damaged by tea mosquito bug"), 4)

    v.add(card("4. Seven-day forecast", ["Give a field's severity and weather history (CSV); an LSTM predicts the next 7 days",
                                         "and warns before the 5% action threshold is crossed.",
                                         "Trained on simulated epidemics: a guide, not a fact."], kicker="For farm managers"), 5)
    v.add(terminal("python -m surveillance.cli analyze leaf.jpg --history demo\\field_history.csv",
                   (DEMO / "cli_forecast.txt").read_text(encoding="utf-8")), 4)
    v.add(caption(forecast_chart(DEMO / "forecast_report" / "leafnet75_00036.json", DEMO / "field_history.csv"),
                  "A true alert from a simulated field. The LSTM catches about 2 in 3 such crossings."), 6)

    v.add(card("5. Video mode", ["Every Nth frame of a scouting video is analysed; annotated frames plus a timeline.",
                                 "SYNTHETIC clip: assembled by panning across still photos, because no openly licensed",
                                 "annotated field video of these diseases exists."], kicker="Scouting walk"), 5)
    timeline = json.loads((DEMO / "video_report" / "field_walk_SYNTHETIC_timeline.json").read_text())
    for row in timeline:
        frame = DEMO / "video_report" / "field_walk_SYNTHETIC_frames" / f"frame_{row['frame']:06d}.jpg"
        v.add(showcase(frame, f"t = {row['seconds']:.0f} s · {row['label']} {row['confidence']:.0%} · severity {row['severity_percent']}%"
                       + (" · alert" if row["alerts"] else "")), 0.9)

    v.add(card("Measured on held-out test sets", results_lines(), kicker="Results", size=23, top=140, wrap=92), 12)
    v.add(card("Use it responsibly", ["An automated first opinion, not professional advice.",
                                      "About 7 in 10 field photos are classified correctly: confirm with an extension officer before spraying.",
                                      "It never decides to spray. Photos are not stored.",
                                      "Full evidence: docs/TECHNICAL_REPORT.md and docs/MODEL_CARD.md"], kicker="Limits"), 7)
    v.close()
    print(f"{OUT} ({v.frames / FPS:.0f} s, {OUT.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
