"""LeafScan Surveillance: the integrated application (brief items 7 and 10).

One photo (or video) in; one report out. Components, each optional and each with its own trained model:
  classify      disease of the leaf            CAFormer-S18 ensemble (the LeafScan web app's model)
  detect        diseased leaves, with boxes    RetinaNet-ResNet50 trained on PlantDoc (13 crops); detector="fasterrcnn"
                                               swaps in Faster R-CNN (more accurate on PlantDoc, about 2x slower)
  pests         pest damage, with boxes        RetinaNet-ResNet50 trained on tea field photos (2 pests);
                                               runs only when the crop is declared as tea (crop="tea"), because
                                               on other crops it mistakes disease spots for pest damage
  segment       damaged tissue, per pixel      DeepLabV3-ResNet50 -> lesion count, severity % and grade 0-5
  forecast      7-day severity outlook         LSTM, from a severity/weather history (CSV), if given

Command line (see docs/USER_GUIDE.md):
    python -m surveillance.cli analyze leaf.jpg --out reports/
    python -m surveillance.cli analyze leaf.jpg --history field_history.csv
    python -m surveillance.cli video field.mp4 --every 15
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "Flask Deployed App"
RUNS = ROOT / "surveillance" / "runs"
sys.path.insert(0, str(APP))

PEST_CROPS = {"tea"}  # crops the pest model was trained on; see docs/MODEL_CARD.md, "out-of-scope use"
DETECT_THRESHOLD = 0.40  # box confidence shown in reports (precision/recall trade-off chosen on validation)
CONFIDENCE_THRESHOLD = 0.80


@dataclass
class Report:
    image: str
    classification: dict = field(default_factory=dict)
    detections: list = field(default_factory=list)
    pests: list = field(default_factory=list)
    segmentation: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_json(self):
        return json.dumps(self.__dict__, indent=1)


class Surveillance:
    def __init__(self, components=("classify", "detect", "pests", "segment", "forecast"), device=None, crop=None,
                 detector="retinanet"):
        self.crop = crop.lower() if crop else None
        self.detector = detector
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(4)
        self.components = []
        if "classify" in components:
            from catalog import load_catalog
            from config import Config
            from model import Predictor
            self.catalog = load_catalog(Config.DISEASE_CSV, Config.SUPPLEMENT_CSV)
            self.classifier = Predictor([(str(APP / "leafscan_hybrid.pt"), 224), (str(APP / "leafscan_hybrid_288.pt"), 288)],
                                        kind="hybrid", require_weights=True, tta=True)
            self.components.append("classify")
        from surveillance import detection
        for comp, data, arch in (("detect", "plantdoc", detector), ("pests", "tea", "retinanet")):
            if comp in components and (RUNS / f"{data}_{arch}" / "best.pt").exists():
                net, names = detection.load(data, arch, self.device)
                setattr(self, f"{comp}_model", (net, names))
                self.components.append(comp)
        if "segment" in components and (RUNS / "leafseg_deeplab" / "best.pt").exists():
            from surveillance import segmentation
            net = segmentation.build_semantic("deeplab", pretrained=False)
            state = torch.load(RUNS / "leafseg_deeplab" / "best.pt", map_location="cpu", weights_only=True)
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
            self.segmenter = net.to(self.device).eval()
            self.components.append("segment")
        if "forecast" in components and (RUNS / "progression.pt").exists():
            from surveillance.progression import Forecaster
            self.forecaster = Forecaster("lstm")
            self.forecaster.load_state_dict(torch.load(RUNS / "progression.pt", map_location="cpu", weights_only=True)["lstm"])
            self.forecaster.eval()
            self.components.append("forecast")

    # ------------------------------------------------------------------ components
    def classify(self, img, report):
        r = self.classifier.predict(img)
        c = self.catalog[r["index"]]
        report.classification = {"label": c.label, "key": c.key, "is_healthy": c.is_healthy, "is_background": c.is_background,
                                 "confidence": round(r["confidence"], 4),
                                 "top3": [{"label": self.catalog[i].label, "confidence": round(p, 4)} for i, p in r["top3"]],
                                 "low_confidence": r["confidence"] < CONFIDENCE_THRESHOLD}
        if r["confidence"] < CONFIDENCE_THRESHOLD:
            report.alerts.append("Classification is uncertain (<80%): retake the photo in daylight, one leaf filling the frame.")

    @torch.inference_mode()
    def _detect(self, img, which):
        from torchvision.transforms import functional as F
        net, names = getattr(self, f"{which}_model")
        out = net([F.to_tensor(img).to(self.device)])[0]
        dets = []
        for box, score, label in zip(out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()):
            if score >= DETECT_THRESHOLD:
                dets.append({"label": names[int(label) - 1], "score": round(float(score), 3),
                             "box_xyxy": [round(float(v), 1) for v in box]})
        return dets

    def detect(self, img, report):
        report.detections = self._detect(img, "detect")

    def pests(self, img, report):
        found = [d for d in self._detect(img, "pests") if "(pest)" in d["label"]]
        report.pests = found
        if found:
            report.alerts.append(f"Possible pest damage: {', '.join(sorted({d['label'] for d in found}))}. "
                                 "Note: the pest model was trained on tea leaves.")

    @torch.inference_mode()
    def segment(self, img, report):
        from torchvision.transforms import functional as F

        from surveillance import segmentation as seg
        small = img.resize((seg.SIZE, seg.SIZE), Image.BILINEAR)
        x = F.normalize(F.to_tensor(small), seg.MEAN, seg.STD)[None].to(self.device)
        mask = self.segmenter(x)["out"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        frac, grade = seg.severity(np.asarray(small), mask)
        report.segmentation = {"lesion_count": len(seg.lesion_instances(mask)), "severity_percent": round(frac * 100, 1),
                               "severity_grade_0_to_5": grade}
        self._mask = mask
        if grade >= 3:
            report.alerts.append(f"Severe damage: about {frac:.0%} of the leaf (grade {grade}/5).")

    def forecast(self, history_csv, report):
        """history_csv columns: day, severity (0-1 or %), temperature_c, humidity_pct, rain_mm, spray (0/1)."""
        from surveillance.progression import HORIZON, LOOKBACK, THRESHOLD
        rows = np.genfromtxt(history_csv, delimiter=",", names=True)
        if len(rows) < LOOKBACK:
            report.notes.append(f"Forecast needs at least {LOOKBACK} days of history; got {len(rows)}.")
            return
        rows = rows[-LOOKBACK:]
        sev = rows["severity"] / (100 if rows["severity"].max() > 1 else 1)
        x = np.stack([sev, (rows["temperature_c"] - 18) / 6, (rows["humidity_pct"] - 70) / 15, rows["rain_mm"] / 10,
                      rows["spray"]], 1).astype(np.float32)
        with torch.no_grad():
            f = self.forecaster(torch.from_numpy(x)[None])[0].numpy()
        report.forecast = {"days_ahead": list(range(1, HORIZON + 1)), "severity_percent": [round(float(v) * 100, 2) for v in f],
                           "crosses_action_threshold": bool(sev[-1] < THRESHOLD <= f.max()),
                           "caveat": "LSTM trained on simulated epidemics (see docs/MODEL_CARD.md); use as a guide only."}
        if report.forecast["crosses_action_threshold"]:
            report.alerts.append(f"Forecast: severity likely to pass {THRESHOLD:.0%} within {HORIZON} days.")

    # ------------------------------------------------------------------ orchestration
    def analyze(self, image, history_csv=None, name="image"):
        img = ImageOps.exif_transpose(image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
        report = Report(image=str(name))
        for comp in self.components:
            if comp == "forecast" and not history_csv:
                continue
            if comp == "classify" and self.crop in PEST_CROPS:
                report.notes.append(f"Disease classification skipped: the classifier was not trained on {self.crop}.")
                continue
            if comp == "pests" and self.crop not in PEST_CROPS:
                report.notes.append("Pest detection skipped: the pest model was trained on tea only and mistakes disease "
                                    "spots on other crops for pests. Use crop='tea' (CLI: --crop tea) for tea photos.")
                continue
            t = time.perf_counter()
            getattr(self, comp)(history_csv if comp == "forecast" else img, report)
            report.timings_ms[comp] = round((time.perf_counter() - t) * 1000, 1)
        report.timings_ms["total"] = round(sum(report.timings_ms.values()), 1)
        missing = {"detect", "pests", "segment"} - set(self.components)
        if missing:
            report.notes.append(f"Components without trained weights were skipped: {', '.join(sorted(missing))}.")
        report.notes.append("Automated first opinion, not professional advice. Confirm with an extension officer.")
        return report, self.render(img, report)

    def render(self, img, report):
        """Annotated copy: lesion overlay (red), disease boxes (blue), pest boxes (orange), summary panel."""
        canvas = img.copy()
        if getattr(self, "_mask", None) is not None and "segment" in self.components:
            m = Image.fromarray(self._mask * 255).resize(canvas.size, Image.NEAREST)
            red = Image.new("RGB", canvas.size, (220, 40, 40))
            canvas = Image.composite(Image.blend(canvas, red, 0.45), canvas, m)
        draw = ImageDraw.Draw(canvas)
        width = max(2, canvas.size[0] // 300)
        size = max(13, canvas.size[0] // 45)
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        for dets, colour in ((report.detections, (42, 120, 214)), (report.pests, (235, 104, 52))):
            for d in dets:
                draw.rectangle(d["box_xyxy"], outline=colour, width=width)
                text = f"{d['label']} {d['score']:.0%}"
                x0, y0 = d["box_xyxy"][0], max(0, d["box_xyxy"][1] - size - 6)
                tw = draw.textlength(text, font=font)
                draw.rectangle([x0, y0, x0 + tw + 8, y0 + size + 6], fill=colour)
                draw.text((x0 + 4, y0 + 2), text, fill=(255, 255, 255), font=font)
        lines = []
        if report.classification:
            c = report.classification
            lines.append(f"{c['label']}  {c['confidence']:.0%}" + ("  (uncertain)" if c["low_confidence"] else ""))
        if report.segmentation:
            s = report.segmentation
            lines.append(f"lesions {s['lesion_count']}, severity {s['severity_percent']}% (grade {s['severity_grade_0_to_5']}/5)")
        if report.pests:
            lines.append(f"pests: {len(report.pests)} detection(s)")
        lines += [f"! {a}" for a in report.alerts[:3]]
        line_h = size + 8
        panel_h = line_h * len(lines) + 14
        out = Image.new("RGB", (canvas.size[0], canvas.size[1] + panel_h), (252, 252, 251))
        out.paste(canvas, (0, 0))
        d2 = ImageDraw.Draw(out)
        for i, line in enumerate(lines):
            d2.text((10, canvas.size[1] + 8 + line_h * i), line[:120], fill=(11, 11, 11), font=font)
        return out

    def analyze_video(self, path, every=15, out_dir=None):
        """Sample every Nth frame, analyse it, and return a per-frame severity timeline."""
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        timeline, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                report, annotated = self.analyze(img, name=f"{Path(path).name}@{idx / fps:.1f}s")
                timeline.append({"frame": idx, "seconds": round(idx / fps, 2), "label": report.classification.get("label"),
                                 "confidence": report.classification.get("confidence"),
                                 "severity_percent": report.segmentation.get("severity_percent"),
                                 "pests": len(report.pests), "alerts": report.alerts})
                if out_dir:
                    annotated.save(Path(out_dir) / f"frame_{idx:06d}.jpg", quality=90)
            idx += 1
        cap.release()
        return timeline
