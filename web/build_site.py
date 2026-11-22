"""Assemble the static browser version of LeafScan (for GitHub Pages) into one folder.

    python web/build_site.py _site                     # exports the ONNX models too (needs torch, timm, onnx)
    python web/build_site.py _site --models DIR        # reuse ONNX files already exported to DIR

The pages are Jinja templates in web/templates/ (base.html is the shared header and footer), rendered
here once: index.html (diagnose in the browser) and treatments.html (all products, with the app's filters).
Everything else (stylesheet, icons, fonts, images, diagnose.js, theme.js, treatments.js, the disease
catalog) is taken from the Flask app, so both versions stay identical.
"""
import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
APP = ROOT / "Flask Deployed App"
STATIC = APP / "static"
sys.path.insert(0, str(APP))

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from catalog import crop_summary, load_catalog  # noqa: E402
from config import Config  # noqa: E402

FROM_APP = ["css/app.css", "js/theme.js", "js/diagnose.js", "js/treatments.js", "icons.svg", "fonts",
            "img/favicon.svg", "img/favicon-32.png", "img/apple-touch-icon.png", "img/og-image.png",
            "img/products", "img/samples"]
FROM_WEB = ["css/web.css", "js/app.js", "js/early.js"]
PAGES = ["index.html", "treatments.html"]
MODEL_FILES = ["leafscan_224.onnx", "leafscan_288.onnx"]


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def render_pages(out: Path, catalog):
    """Render the static pages. Autoescaping is on, so text from the CSVs can never become HTML."""
    env = Environment(loader=FileSystemLoader(WEB / "templates"), autoescape=True,
                      trim_blocks=True, lstrip_blocks=True)
    crops = crop_summary(catalog)
    context = {
        "products": [c for c in catalog if not c.is_background and c.product_link],
        "crops": crops,
        "stats": {"diseases": sum(not (c.is_healthy or c.is_background) for c in catalog), "crops": len(crops)},
    }
    for name in PAGES:
        (out / name).write_text(env.get_template(name).render(**context), encoding="utf-8")


def write_catalog(out: Path, catalog):
    """The 39 conditions in the models' class order, with the text the result page shows."""
    rows = []
    for c in catalog:
        row = dataclasses.asdict(c)
        row.pop("ref_image")  # hot-linked third-party photos are not used on the static site
        row.update(label=c.label, product_kind=c.product_kind)
        rows.append(row)
    (out / "catalog.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", type=Path, help="folder with already exported ONNX files")
    args = ap.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for rel in FROM_APP:
        copy(STATIC / rel, out / rel)
    for rel in FROM_WEB:
        copy(WEB / rel, out / rel)
    catalog = load_catalog(Config.DISEASE_CSV, Config.SUPPLEMENT_CSV)
    render_pages(out, catalog)
    n = write_catalog(out, catalog)
    (out / ".nojekyll").write_text("")  # serve files as they are (no Jekyll processing on GitHub Pages)

    models = out / "models"
    if args.models:
        for name in MODEL_FILES:
            copy(args.models / name, models / name)
    else:
        sys.path.insert(0, str(WEB))
        from export_onnx import MODELS, export
        models.mkdir(parents=True, exist_ok=True)
        for name in MODELS:
            export(name, models)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
    print(f"Site written to {out}: {n} conditions, {size:.0f} MB in total")


if __name__ == "__main__":
    main()
