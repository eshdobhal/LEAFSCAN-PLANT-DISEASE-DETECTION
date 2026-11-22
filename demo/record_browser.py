"""Record the live browser version (GitHub Pages) for the demo video: diagnose an uploaded photo, then the
Treatments page. Writes demo/browser/*.webm plus marks.json (when the model finished loading, so the
assembly script can cut the one-time download). Run with the web environment: .venv/Scripts/python."""
import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "https://eshdobhal.github.io/LEAFSCAN-PLANT-DISEASE-DETECTION/"
OUT = ROOT / "demo" / "browser"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.webm"):
        f.unlink()
    marks = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(OUT),
                                  record_video_size={"width": 1280, "height": 720})
        t0 = time.monotonic()
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_function("window.LeafScan !== undefined && window.LeafScan.ready", timeout=300000)
        marks["model_ready"] = time.monotonic() - t0
        page.wait_for_timeout(1500)
        page.set_input_files("#image", str(ROOT / "demo" / "photos" / "leafnet75_00063.jpg"))
        page.wait_for_timeout(1500)
        page.click("[data-submit]")
        expect(page.locator("#result-title")).to_contain_text("blight", timeout=60000)
        marks["result"] = time.monotonic() - t0
        page.wait_for_timeout(2500)
        for _ in range(6):
            page.mouse.wheel(0, 220)
            page.wait_for_timeout(700)
        page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1200)
        page.click("#site-nav a[href='treatments.html']")
        expect(page.locator("h1")).to_have_text("Treatments & fertilisers")
        page.wait_for_timeout(1500)
        page.click("button[data-crop=tomato]")
        page.wait_for_timeout(1500)
        page.locator("input[data-search]").press_sequentially("blight", delay=150)
        page.wait_for_timeout(3000)
        marks["end"] = time.monotonic() - t0
        ctx.close()
        browser.close()
    (OUT / "marks.json").write_text(json.dumps(marks, indent=1))
    print(marks, [f.name for f in OUT.glob("*.webm")])


if __name__ == "__main__":
    main()
