"""Command-line interface of LeafScan Surveillance.

    python -m surveillance.cli analyze PHOTO [PHOTO ...] [--out DIR] [--history CSV] [--only classify,segment] [--crop tea]
    python -m surveillance.cli video VIDEO [--every N] [--out DIR]
    python -m surveillance.cli status

`analyze` writes <name>.json (the full report) and <name>_annotated.png into --out (default: reports/).
Exit code 0 = no alerts, 2 = at least one alert (useful in scripts and schedulers), 1 = error.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="leafscan-surveillance", description="Crop disease and pest surveillance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse one or more photos")
    a.add_argument("photos", nargs="+", type=Path)
    a.add_argument("--out", type=Path, default=Path("reports"))
    a.add_argument("--history", type=Path, help="CSV of past severity and weather, for the 7-day forecast")
    a.add_argument("--only", default="classify,detect,pests,segment,forecast", help="comma-separated components")
    a.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    a.add_argument("--crop", default=None, help="crop in the photos; pest detection runs only for --crop tea")
    a.add_argument("--detector", default="retinanet", choices=["retinanet", "fasterrcnn"],
                   help="disease detector: retinanet (default, faster) or fasterrcnn (more accurate on PlantDoc)")
    v = sub.add_parser("video", help="analyse every Nth frame of a video")
    v.add_argument("video", type=Path)
    v.add_argument("--every", type=int, default=15)
    v.add_argument("--out", type=Path, default=Path("reports"))
    v.add_argument("--device", default=None)
    v.add_argument("--crop", default=None)
    v.add_argument("--detector", default="retinanet", choices=["retinanet", "fasterrcnn"])
    sub.add_parser("status", help="which components have trained weights")
    args = ap.parse_args(argv)

    from surveillance.pipeline import Surveillance
    if args.cmd == "status":
        s = Surveillance(device="cpu")
        print(json.dumps({"available_components": s.components}, indent=1))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        s = Surveillance(tuple(args.only.split(",")), device=args.device, crop=args.crop, detector=args.detector)
        any_alert = False
        for photo in args.photos:
            report, annotated = s.analyze(photo, history_csv=args.history, name=photo.name)
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
            any_alert |= bool(report.alerts)
        return 2 if any_alert else 0
    if args.cmd == "video":
        s = Surveillance(device=args.device, crop=args.crop, detector=args.detector)
        frames = args.out / f"{args.video.stem}_frames"
        frames.mkdir(parents=True, exist_ok=True)
        timeline = s.analyze_video(args.video, args.every, frames)
        (args.out / f"{args.video.stem}_timeline.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
        print(f"{len(timeline)} frames analysed; timeline in {args.out / (args.video.stem + '_timeline.json')}")
        return 2 if any(t["alerts"] for t in timeline) else 0


if __name__ == "__main__":
    sys.exit(main())
