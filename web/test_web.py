"""Browser tests for the static LeafScan site (Playwright + Chromium).

    python web/test_web.py SITE_DIR            # smoke test: load, sample, upload, no console/CSP errors
    python web/test_web.py SITE_DIR --parity   # also: every field test photo, browser vs PyTorch (needs data/)

SITE_DIR is the folder made by build_site.py. It is served under /LEAFSCAN-PLANT-DISEASE-DETECTION/ like
GitHub Pages does, so relative paths are tested too.
"""
import argparse
import base64
import functools
import http.server
import json
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "LEAFSCAN-PLANT-DISEASE-DETECTION"


class SubPathHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the site folder at /BASE/ (like username.github.io/BASE/), quietly."""

    def translate_path(self, path):
        prefix = "/" + BASE
        return super().translate_path(path[len(prefix):] if path.startswith(prefix + "/") else "/__outside__")

    def log_message(self, *args):
        pass


def serve(site: Path) -> str:
    handler = functools.partial(SubPathHandler, directory=str(site.resolve()))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/{BASE}/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", type=Path)
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()
    from playwright.sync_api import expect, sync_playwright

    url = serve(args.site)
    failures = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("window.LeafScan !== undefined")
        ok = page.evaluate("window.LeafScan.ready")
        print("models loaded:", ok, "| backend:", page.evaluate("window.LeafScan.backend()"))
        if not ok:
            failures.append("models did not load")

        # A sample photo, clicked like a user would.
        page.click("[data-sample=tomato-leaf-curl]")
        expect(page.locator("#result-title")).to_contain_text("Yellow leaf curl virus", timeout=60000)
        print("sample:", page.locator("#result-title").inner_text().replace("\n", " "), "|",
              page.locator(".verdict__confidence").inner_text())

        # An uploaded photo, through the real form (diagnose.js resizes it first).
        page.set_input_files("#image", str(ROOT / "test_images" / "potato_early_blight.JPG"))
        expect(page.locator("[data-preview]")).to_be_visible()
        page.click("[data-submit]")
        expect(page.locator("#result-title")).to_contain_text("Early blight", timeout=60000)
        expect(page.locator("[data-submit]")).to_be_enabled()  # loading state reset for the next photo
        print("upload:", page.locator("#result-title").inner_text().replace("\n", " "))

        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        if overflow > 0:
            failures.append(f"horizontal overflow {overflow}px")

        # Treatments page: reached from the nav, every product listed, the app's filters work.
        treatments = browser.new_page(viewport={"width": 1280, "height": 900})
        treatments.on("console", lambda m: m.type == "error" and errors.append(m.text))
        treatments.on("pageerror", lambda e: errors.append(str(e)))
        treatments.goto(url)
        treatments.click("#site-nav a[href='treatments.html']")
        expect(treatments.locator("h1")).to_have_text("Treatments & fertilisers")
        expect(treatments.locator("#site-nav a[aria-current=page]")).to_have_text("Treatments")
        cards = treatments.locator("[data-products] > .product:visible")
        total = cards.count()
        treatments.click("[data-kind=fertiliser]")
        treatments.click("[data-crop=tomato]")
        expect(cards).to_have_count(1)
        treatments.click("[data-kind=all]")
        treatments.click("[data-crop=all]")
        treatments.fill("[data-search]", "blight")
        expect(cards).to_have_count(6)
        expect(treatments.locator("[data-count]")).to_have_text("6 products")
        treatments.fill("[data-search]", "zzz")
        expect(treatments.locator("[data-empty]")).to_be_visible()
        print(f"treatments: {total} products; Fertilisers + Tomato = 1, 'blight' = 6, empty state shown")
        for width in (1280, 375, 320):
            treatments.set_viewport_size({"width": width, "height": 800})
            over = treatments.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            if over > 0:
                failures.append(f"treatments overflows by {over}px at {width}px")
        if total != 38:
            failures.append(f"expected 38 products, found {total}")

        if errors:
            failures.append("console errors: " + " | ".join(errors[:5]))

        if args.parity:
            import numpy as np
            splits = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))["test_field"]
            torch_pred = np.load(ROOT / "training" / "runs" / "ensemble" / "test_predictions.npz")["test_field_pred"]
            agree = correct = 0
            for i, (path, label, _) in enumerate(splits):
                data = base64.b64encode((ROOT / "data" / path).read_bytes()).decode()
                index = page.evaluate("""async (b64) => {
                    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                    const bitmap = await createImageBitmap(new Blob([bytes]));
                    return (await window.LeafScan.diagnoseBitmap(bitmap)).index;
                }""", data)
                agree += int(index == int(torch_pred[i]))
                correct += int(index == label)
            n = len(splits)
            print(f"parity on {n} field test photos: browser = PyTorch on {agree}/{n} ({agree / n:.1%}); "
                  f"browser accuracy {correct / n:.2%} (PyTorch ensemble 71.11%)")
        browser.close()

    print("FAILED: " + "; ".join(failures) if failures else "All browser checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
