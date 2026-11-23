"""Command-line interface of LeafScan Surveillance.

    python -m surveillance.cli analyze PHOTO [PHOTO ...] [--out DIR] [--history CSV] [--only classify,segment] [--crop tea]
    python -m surveillance.cli video VIDEO [--every N] [--out DIR]
    python -m surveillance.cli status
    (any command) [--log-file run.log] [--verbose]

`analyze` writes <name>.json (the full report) and <name>_annotated.png into --out (default: reports/), plus
run_config.json (every setting of the run: components, device, crop, detector, thresholds, versions).
Exit code 0 = no alerts, 2 = at least one alert (useful in scripts and schedulers), 1 = an error (for example an
unreadable photo; the other photos are still analysed and the error is logged).
"""
import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

log = logging.getLogger("leafscan.surveillance")


def setup_logging(log_file=None, verbose=False):
    """Warnings and errors to the console (info too with --verbose); everything, timestamped, to --log-file."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        f = logging.FileHandler(log_file, encoding="utf-8")
        f.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(f)


def run_config(s, args):
    import torch

    from surveillance.pipeline import CONFIDENCE_THRESHOLD, DETECT_THRESHOLD, THRESHOLDS
    return {"command": args.cmd, "components": s.components, "device": str(s.device), "crop": s.crop,
            "detector": s.detector, "detection_thresholds": THRESHOLDS or {"default": DETECT_THRESHOLD},
            "classifier_not_sure_below": CONFIDENCE_THRESHOLD, "cpu_threads": torch.get_num_threads(),
            "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-file", type=Path, help="also write a timestamped log of the run to this file")
    common.add_argument("--verbose", "-v", action="store_true", help="show progress (info messages) on the console")
    ap = argparse.ArgumentParser(prog="leafscan-surveillance", description="Crop disease and pest surveillance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", parents=[common], help="analyse one or more photos")
    a.add_argument("photos", nargs="+", type=Path)
    a.add_argument("--out", type=Path, default=Path("reports"))
    a.add_argument("--history", type=Path, help="CSV of past severity and weather, for the 7-day forecast")
    a.add_argument("--only", default="classify,detect,pests,segment,forecast", help="comma-separated components")
    a.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    a.add_argument("--crop", default=None, help="crop in the photos; pest detection runs only for --crop tea")
    a.add_argument("--detector", default="retinanet", choices=["retinanet", "fasterrcnn"],
                   help="disease detector: retinanet (default, faster) or fasterrcnn (more accurate on PlantDoc)")
    v = sub.add_parser("video", parents=[common], help="analyse every Nth frame of a video")
    v.add_argument("video", type=Path)
    v.add_argument("--every", type=int, default=15)
    v.add_argument("--out", type=Path, default=Path("reports"))
    v.add_argument("--device", default=None)
    v.add_argument("--crop", default=None)
    v.add_argument("--detector", default="retinanet", choices=["retinanet", "fasterrcnn"])
    sub.add_parser("status", parents=[common], help="which components have trained weights")
    args = ap.parse_args(argv)
    setup_logging(args.log_file, args.verbose)

    from surveillance.pipeline import Surveillance
    if args.cmd == "status":
        s = Surveillance(device="cpu")
        print(json.dumps({"available_components": s.components}, indent=1))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        s = Surveillance(tuple(args.only.split(",")), device=args.device, crop=args.crop, detector=args.detector)
        config = run_config(s, args)
        (args.out / "run_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        log.info("run %s", json.dumps(config))
        any_alert = errors = 0
        for photo in args.photos:
            try:
                report, annotated = s.analyze(photo, history_csv=args.history, name=photo.name)
            except Exception:  # one bad photo must not stop a batch; it is logged and reflected in the exit code
                log.exception("failed: %s", photo)
                errors += 1
                continue
            (args.out / f"{photo.stem}.json").write_text(report.to_json(), encoding="utf-8")
            annotated.save(args.out / f"{photo.stem}_annotated.png")
            c = report.classification
            print(f"{photo.name}: " + (f"{c['label']} ({c['confidence']:.0%})" if c else "not classified")
                  + (f" | severity {report.segmentation['severity_percent']}%" if report.segmentation else "")
                  + (f" | pests {len(report.pests)}" if "pests" in s.components and s.crop == "tea" else "")
                  + (f" | 7-day forecast max {max(report.forecast['severity_percent'])}%" if report.forecast else "")
                  + f" | {report.timings_ms.get('total')} ms")
            for alert in report.alerts:
                print(f"  ! {alert}")
            log.info("photo %s", json.dumps({"photo": str(photo), "label": c.get("label"), "confidence": c.get("confidence"),
                                              "severity_percent": (report.segmentation or {}).get("severity_percent"),
                                              "pests": len(report.pests), "alerts": report.alerts, "timings_ms": report.timings_ms}))
            any_alert |= bool(report.alerts)
        log.info("done: %d photo(s), %d error(s), alerts=%s", len(args.photos), errors, bool(any_alert))
        return 1 if errors else (2 if any_alert else 0)
    if args.cmd == "video":
        s = Surveillance(device=args.device, crop=args.crop, detector=args.detector)
        config = run_config(s, args)
        (args.out / "run_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        log.info("run %s", json.dumps(config))
        frames = args.out / f"{args.video.stem}_frames"
        frames.mkdir(parents=True, exist_ok=True)
        try:
            timeline = s.analyze_video(args.video, args.every, frames)
        except Exception:
            log.exception("failed: %s", args.video)
            return 1
        (args.out / f"{args.video.stem}_timeline.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
        print(f"{len(timeline)} frames analysed; timeline in {args.out / (args.video.stem + '_timeline.json')}")
        log.info("done: %d frame(s) analysed", len(timeline))
        return 2 if any(t["alerts"] for t in timeline) else 0


if __name__ == "__main__":
    sys.exit(main())
