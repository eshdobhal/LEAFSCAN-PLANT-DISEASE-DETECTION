"""Full-page screenshots of every LeafScan page at 375 px and 1440 px, light and dark.

    .venv\\Scripts\\python scripts\\screenshots.py

Starts the app with waitress on a free port, drives it with Playwright's Chromium,
and writes PNGs to screenshots/. Also prints console errors and any horizontal overflow.
"""
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "Flask Deployed App"
OUT = ROOT / "screenshots"
sys.path.insert(0, str(APP_DIR))

from playwright.sync_api import sync_playwright  # noqa: E402
from waitress import create_server  # noqa: E402

from app import create_app  # noqa: E402

SAMPLE = ROOT / "test_images" / "tomato_early_blight.JPG"
WIDTHS = {"mobile": 375, "desktop": 1440}
PAGES = {
    "home": "/",
    "diagnose": "/diagnose",
    "treatments": "/treatments",
    "about": "/about",
    "404": "/no-such-page",
}


def start_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = create_server(create_app(), host="127.0.0.1", port=port, threads=8)
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(0.5)
    return f"http://127.0.0.1:{port}"


def check_page(page, name, problems):
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    if overflow > 0:
        problems.append(f"{name}: horizontal overflow of {overflow}px")


def main():
    OUT.mkdir(exist_ok=True)
    base = start_server()
    problems = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for scheme in ("light", "dark"):
            for label, width in WIDTHS.items():
                context = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme,
                                              reduced_motion="reduce")
                page = context.new_page()
                page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
                for name, path in PAGES.items():
                    page.goto(base + path)
                    page.wait_for_load_state("networkidle")
                    check_page(page, f"{name}-{label}-{scheme}", problems)
                    page.screenshot(path=OUT / f"{name}-{label}-{scheme}.png", full_page=True)

                # Diagnose with a photo chosen, then the result page.
                page.goto(base + "/diagnose")
                page.set_input_files("#image", str(SAMPLE))
                page.wait_for_selector("[data-preview]:not([hidden])")
                page.screenshot(path=OUT / f"diagnose-selected-{label}-{scheme}.png", full_page=True)
                page.click("[data-submit]")
                page.wait_for_url("**/diagnose")
                page.wait_for_load_state("networkidle")
                check_page(page, f"result-{label}-{scheme}", problems)
                page.screenshot(path=OUT / f"result-{label}-{scheme}.png", full_page=True)

                if label == "mobile":  # open phone menu
                    page.goto(base + "/")
                    page.click(".menu-toggle")
                    page.screenshot(path=OUT / f"menu-open-{label}-{scheme}.png")
                context.close()

        # Extra outputs for the README (desktop, both themes): low-confidence result, healthy-leaf sample,
        # a field-photo sample, and the camera dialog with Chromium's fake camera.
        for scheme in ("light", "dark"):
            context = browser.new_context(viewport={"width": 1280, "height": 900}, color_scheme=scheme,
                                          reduced_motion="reduce")
            page = context.new_page()
            page.goto(base + "/diagnose")
            page.set_input_files("#image", str(ROOT / "test_images" / "tomato-mold.jpg"))
            page.wait_for_selector("[data-preview]:not([hidden])")
            page.click("[data-submit]")
            page.wait_for_url("**/diagnose")
            page.screenshot(path=OUT / f"result-low-confidence-{scheme}.png")
            for sample in ("tomato-healthy", "tomato-leaf-curl"):
                page.goto(base + "/diagnose")
                page.click(f".sample[value={sample}]")
                page.wait_for_url("**/diagnose/sample")
                page.screenshot(path=OUT / f"result-sample-{sample}-{scheme}.png")
            context.close()

        camera_browser = pw.chromium.launch(args=["--use-fake-ui-for-media-stream",
                                                  "--use-fake-device-for-media-stream"])
        # bypass_csp: Playwright's wait_for_function evaluates code, which our strict CSP (rightly) blocks.
        context = camera_browser.new_context(viewport={"width": 1280, "height": 900}, permissions=["camera"],
                                             reduced_motion="reduce", bypass_csp=True)
        page = context.new_page()
        page.goto(base + "/diagnose")
        page.click("[data-camera-open]")
        page.wait_for_function("document.querySelector('[data-camera-video]').readyState >= 2")
        page.wait_for_timeout(500)
        page.screenshot(path=OUT / "camera-dialog-light.png")
        camera_browser.close()

        # 320 px: the narrowest phone we support must not scroll sideways.
        context = browser.new_context(viewport={"width": 320, "height": 700})
        page = context.new_page()
        for name, path in PAGES.items():
            page.goto(base + path)
            check_page(page, f"{name}-320", problems)
        browser.close()

    print("Screenshots written to", OUT)
    print("\n".join(problems) if problems else "No console errors or horizontal overflow found.")


if __name__ == "__main__":
    main()
