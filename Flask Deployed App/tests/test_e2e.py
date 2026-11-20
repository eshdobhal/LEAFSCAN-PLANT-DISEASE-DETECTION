"""End-to-end tests in a real browser (Playwright + Chromium): what a user actually clicks.

Skipped unless E2E=1, because they need Chromium (python -m playwright install chromium) and take ~1 min:

    $env:E2E = "1"; python -m pytest tests/test_e2e.py; Remove-Item Env:E2E
"""
import io
import os
import re
import socket
import threading

import pytest
from PIL import Image

from conftest import TEST_IMAGES

pytestmark = pytest.mark.skipif(os.environ.get("E2E") != "1", reason="browser tests: set E2E=1 to run")

if os.environ.get("E2E") == "1":
    from playwright.sync_api import expect, sync_playwright
    from waitress import create_server

# Reference photos are hot-linked from third-party sites; some block that, which is expected (hidden on error).
IGNORED_CONSOLE = ("ERR_BLOCKED_BY_RESPONSE", "ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION", "ERR_CERT")


@pytest.fixture(scope="module")
def base_url(app):
    """Serve the already-loaded app with waitress on a free port, in a background thread."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = create_server(app, host="127.0.0.1", port=port, threads=4)
    threading.Thread(target=server.run, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.close()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        # Fake camera so the camera dialog can be tested without hardware or a permission prompt.
        b = pw.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])
        yield b
        b.close()


@pytest.fixture()
def page(browser, base_url):
    context = browser.new_context(base_url=base_url, viewport={"width": 1280, "height": 900},
                                  permissions=["camera"], reduced_motion="reduce")
    p = context.new_page()
    p.console_errors = []
    p.on("console", lambda m: m.type == "error" and not any(x in m.text for x in IGNORED_CONSOLE)
         and p.console_errors.append(m.text))
    yield p
    context.close()


def diagnose_with(page, path):
    page.goto("/diagnose")
    page.set_input_files("#image", str(path))
    expect(page.locator("[data-preview]")).to_be_visible()
    with page.expect_navigation():
        page.click("[data-submit]")


def test_upload_shows_diagnosis(page, model_loaded):
    diagnose_with(page, TEST_IMAGES / "tomato_early_blight.JPG")
    expect(page.locator("h1")).to_contain_text("Early blight" if model_loaded else "")
    expect(page.locator(".bars__row")).to_have_count(3)
    expect(page.locator(".verdict__photo")).to_have_attribute("src", re.compile(r"^data:image/jpeg;base64,"))
    assert page.console_errors == []


def test_sample_button_runs_a_diagnosis(page, model_loaded):
    page.goto("/diagnose")
    with page.expect_navigation():
        page.click(".sample[value=peach-bacterial-spot]")
    expect(page.locator(".eyebrow").first).to_contain_text("sample photo")
    expect(page.locator("h1")).to_contain_text("Bacterial spot" if model_loaded else "")
    assert page.console_errors == []


def test_large_png_is_resized_in_browser(page, tmp_path):
    big = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), (40, 120, 60)).save(big)
    page.goto("/diagnose")
    page.set_input_files("#image", str(big))
    expect(page.locator("[data-preview]")).to_be_visible()
    info = page.evaluate("""async () => {
        const f = document.querySelector('#image').files[0];
        const bmp = await createImageBitmap(f);
        return {type: f.type, w: bmp.width, h: bmp.height};
    }""")
    assert info == {"type": "image/jpeg", "w": 1024, "h": 768}


def test_non_image_shows_inline_error(page, tmp_path):
    fake = tmp_path / "notes.txt"
    fake.write_text("not a photo")
    page.goto("/diagnose")
    page.set_input_files("#image", str(fake))
    expect(page.locator("#upload-error")).not_to_be_empty()
    expect(page.locator("[data-preview]")).to_be_hidden()


def test_camera_capture_fills_the_form(page):
    page.goto("/diagnose")
    page.click("[data-camera-open]")
    expect(page.locator("[data-camera]")).to_be_visible()
    page.click("[data-camera-capture]")
    page.click("[data-camera-use]")
    expect(page.locator("[data-camera]")).to_be_hidden()
    assert page.evaluate("document.querySelector('#image').files[0].name") == "camera-photo.jpg"
    expect(page.locator("[data-preview]")).to_be_visible()
    # The camera must be switched off once the dialog closes.
    assert page.evaluate("[...document.querySelectorAll('video')].every(v => !v.srcObject)")


def test_treatment_filters(page):
    page.goto("/treatments")
    total = page.locator("[data-products] > .product").count()
    page.click("[data-kind=fertiliser]")
    page.click("[data-crop=tomato]")
    expect(page.locator("[data-products] > .product:visible")).to_have_count(1)
    page.click("[data-kind=all]")
    page.fill("[data-search]", "blight")  # tomato early + late blight
    expect(page.locator("[data-products] > .product:visible")).to_have_count(2)
    page.click("[data-crop=all]")  # + corn, grape and both potato blights
    expect(page.locator("[data-products] > .product:visible")).to_have_count(6)
    expect(page.locator("[data-count]")).to_have_text("6 products")
    page.fill("[data-search]", "zzz")
    expect(page.locator("[data-empty]")).to_be_visible()
    page.fill("[data-search]", "")
    expect(page.locator("[data-products] > .product:visible")).to_have_count(total)


def test_theme_toggle_persists(page):
    page.goto("/")
    page.click(".theme-toggle")
    theme = page.evaluate("document.documentElement.dataset.theme")
    assert theme in ("light", "dark")
    page.reload()
    assert page.evaluate("document.documentElement.dataset.theme") == theme


def test_mobile_menu(browser, base_url):
    context = browser.new_context(base_url=base_url, viewport={"width": 375, "height": 800})
    page = context.new_page()
    page.goto("/")
    toggle = page.locator(".menu-toggle")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    expect(page.locator("#site-nav")).to_be_hidden()
    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(page.locator("#site-nav")).to_be_visible()
    page.keyboard.press("Escape")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    context.close()


def test_works_without_javascript(browser, base_url, model_loaded):
    context = browser.new_context(base_url=base_url, java_script_enabled=False)
    page = context.new_page()
    page.goto("/diagnose")
    expect(page.locator("#site-nav")).to_be_visible()
    page.set_input_files("#image", str(TEST_IMAGES / "potato_late_blight.JPG"))
    with page.expect_navigation():
        page.click("[data-submit]")
    expect(page.locator("h1")).to_contain_text("Late blight" if model_loaded else "")
    context.close()


@pytest.mark.parametrize("width", [320, 375, 1440])
def test_no_horizontal_overflow(browser, base_url, width):
    context = browser.new_context(base_url=base_url, viewport={"width": width, "height": 800})
    page = context.new_page()
    for path in ("/", "/diagnose", "/treatments", "/about", "/missing-page"):
        page.goto(path)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0, f"{path} overflows by {overflow}px at {width}px"
    context.close()


def test_keyboard_skip_link_is_first(page):
    page.goto("/")
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.className") == "skip-link"
    page.keyboard.press("Enter")
    assert page.evaluate("location.hash") == "#main"


def test_pages_have_no_console_or_csp_errors(page):
    for path in ("/", "/diagnose", "/treatments", "/about"):
        page.goto(path)
    assert page.console_errors == []


def test_api_from_browser_context(page):
    """The JSON API, called the way another app would (multipart POST)."""
    buf = io.BytesIO()
    Image.open(TEST_IMAGES / "corn_common_rust.JPG").save(buf, "JPEG")
    response = page.request.post("/api/predict", multipart={
        "image": {"name": "leaf.jpg", "mimeType": "image/jpeg", "buffer": buf.getvalue()}})
    assert response.ok
    data = response.json()
    assert {"label", "confidence", "top3", "low_confidence", "demo_mode"} <= data.keys()
