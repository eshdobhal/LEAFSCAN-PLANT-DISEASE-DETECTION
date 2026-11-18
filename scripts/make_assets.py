"""Regenerates LeafScan's artwork: crop/product/hero/favicon SVGs, PNG icons and the Open Graph image.

    .venv\\Scripts\\python scripts\\make_assets.py

SVGs are written directly. PNGs are rendered from the SVGs / an HTML card with Playwright's Chromium,
so no image editor or third-party art is involved.
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "Flask Deployed App" / "static"
IMG = STATIC / "img"

INK, LEAF, FRUIT, BADGE, PAPER = "#1F5135", "#7FB24A", "#D7E8C0", "#E4EEE6", "#F7F5EF"


# ---------- SVG building blocks ----------
def c(x, y, r, fill=FRUIT):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>'


def p(d, fill=FRUIT):
    return f'<path d="{d}" fill="{fill}"/>'


def dot(x, y, r=1.4):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{INK}" stroke="none"/>'


def line(d):
    return f'<path d="{d}" fill="none"/>'


# One flat illustration per crop, 64x64, same 2.5 px stroke, brand colours only.
CROPS = {
    "apple": [p("M32 22c-4-3-12-4-16 2-5 7-2 20 6 24 3 2 6 2 10 0 4 2 7 2 10 0 8-4 11-17 6-24-4-6-12-5-16-2z"),
              line("M32 22c0-4 1-7 3-9"), p("M35 16c3-4 8-5 11-4-1 4-6 6-11 4z", LEAF)],
    "blueberry": [line("M32 16c2-4 6-6 10-6"), p("M40 11c3-3 8-3 10-1-2 3-7 4-10 1z", LEAF),
                  c(32, 26, 9), c(23, 41, 9), c(41, 41, 9),
                  c(32, 26, 2.2, "none"), c(23, 41, 2.2, "none"), c(41, 41, 2.2, "none")],
    "cherry": [line("M22 37C24 27 30 19 36 14"), line("M42 35C41 27 39 20 36 14"),
               p("M36 14c4-4 10-4 13-2-2 4-8 5-13 2z", LEAF), c(22, 44, 8), c(42, 42, 8)],
    "corn": [p("M32 11c-7 0-10 10-10 22s4 19 10 19 10-7 10-19-3-22-10-22z"),
             line("M32 12v39M24 22h16M23 29h18M23 36h18M24 43h16"),
             p("M23 33c-7 4-9 13-7 21 5-2 11-3 16-3-6-4-9-10-9-18z", LEAF),
             p("M41 33c7 4 9 13 7 21-5-2-11-3-16-3 6-4 9-10 9-18z", LEAF)],
    "grape": [line("M32 19v-7"), p("M32 15c-4-4-10-4-13-1 3 4 9 4 13 1z", LEAF),
              c(26, 25, 5.5), c(38, 25, 5.5), c(20, 35, 5.5), c(32, 35, 5.5), c(44, 35, 5.5),
              c(26, 45, 5.5), c(38, 45, 5.5), c(32, 54, 5.5)],
    "orange": [c(32, 37, 15), line("M32 22v-4"), p("M32 21c2-5 7-8 12-7-1 5-6 8-12 7z", LEAF),
               line("M24 32a9 9 0 0 1 5-5"), dot(29, 41), dot(36, 44), dot(38, 37)],
    "peach": [p("M32 22c-9-2-17 4-17 14 0 9 8 15 17 15s17-6 17-15c0-10-8-16-17-14z"),
              line("M32 22c-4 8-4 20 0 29"), p("M33 21c3-5 9-7 13-5-2 5-8 7-13 5z", LEAF)],
    "pepper": [p("M22 24c-6 2-8 10-6 18 2 7 6 10 10 10 2 0 4-1 6-2 2 1 4 2 6 2 4 0 8-3 10-10 2-8 0-16-6-18"
                 "-3-1-7 0-10 2-3-2-7-3-10-2z"),
               line("M27 31c-1 6 0 12 2 18M37 31c1 6 0 12-2 18"), p("M25 24c2 2 5 2 7 0 2 2 5 2 7 0-2-3-5-4-7-3-2-1-5 0-7 3z", LEAF),
               line("M32 23c0-5 2-9 6-11")],
    "potato": [p("M17 31c2-9 12-14 23-12 9 2 13 9 11 17-2 9-11 14-21 13-9-1-15-9-13-18z"),
               dot(25, 30), dot(38, 26), dot(42, 39), dot(29, 42)],
    "raspberry": [line("M32 19v-6"), p("M23 21c3 2 6 2 9-1 3 3 6 3 9 1-3-4-6-5-9-3-3-2-6-1-9 3z", LEAF),
                  c(24, 28, 4.5), c(32, 28, 4.5), c(40, 28, 4.5), c(20, 36, 4.5), c(28, 36, 4.5),
                  c(36, 36, 4.5), c(44, 36, 4.5), c(24, 44, 4.5), c(32, 44, 4.5), c(40, 44, 4.5), c(32, 52, 4.5)],
    "soybean": [p("M13 43c2-13 13-24 28-27 6-1 11 2 10 8-2 11-15 23-30 25-5 1-9-1-8-6z", LEAF),
                c(22, 41, 4.5), c(31, 33, 4.5), c(40, 25, 4.5), line("M49 18c2-3 4-5 7-5")],
    "squash": [p("M27 14c-4 0-5 4-5 9 0 6 2 9-1 14-3 5-6 9-4 14 2 6 8 8 15 8s13-2 15-8c2-5-1-9-4-14"
                 "-3-5-1-8-1-14 0-5-1-9-5-9z"),
               line("M32 14V9"), line("M22 41c6 3 14 3 20 0")],
    "strawberry": [p("M16 27c0-4 6-6 16-6s16 2 16 6c0 10-8 24-16 26-8-2-16-16-16-26z"),
                   p("M19 24c4-6 9-6 13-2 4-4 9-4 13 2-4 2-9 2-13 0-4 2-9 2-13 0z", LEAF), line("M32 21v-7"),
                   dot(25, 31), dot(39, 31), dot(32, 36), dot(24, 40), dot(40, 40), dot(32, 46)],
    "tomato": [p("M32 20c-10 0-18 7-18 16s8 16 18 16 18-7 18-16-8-16-18-16z"),
               p("M21 23c4 0 8-1 11-4 3 3 7 4 11 4-3 3-7 3-11 2-4 1-8 1-11-2z", LEAF), line("M32 19v-6"),
               line("M22 34a10 10 0 0 1 5-6")],
}


def crop_svg(parts):
    body = "\n    ".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">\n'
            f'  <circle cx="32" cy="32" r="32" fill="{BADGE}"/>\n'
            f'  <g stroke="{INK}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">\n'
            f'    {body}\n  </g>\n</svg>\n')


TREATMENT_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
  <rect width="96" height="96" rx="16" fill="{BADGE}"/>
  <g stroke="{INK}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <path d="M40 22h14v10H40z" fill="{LEAF}"/>
    <path d="M43 22v-6h14l8 5" fill="none"/>
    <path d="M34 40c0-5 4-8 8-8h10c4 0 8 3 8 8v36c0 3-2 5-5 5H39c-3 0-5-2-5-5z" fill="#FFFFFF"/>
    <path d="M34 48h26v22H34z" fill="{FRUIT}"/>
    <path d="M47 51c-5 2-7 7-6 11 1 3 3 5 6 5s5-2 6-5c1-4-1-9-6-11z" fill="{LEAF}" stroke-width="2"/>
    <path d="M47 56v10" stroke-width="2"/>
  </g>
</svg>
"""

FERTILISER_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
  <rect width="96" height="96" rx="16" fill="#F4EBD7"/>
  <g stroke="#6B4A12" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <path d="M30 30c0-4 4-8 8-8h20c4 0 8 4 8 8l5 42c0 4-3 6-6 6H31c-3 0-6-2-6-6z" fill="#FFFFFF"/>
    <path d="M34 26c8 4 20 4 28 0" fill="none"/>
    <path d="M33 44h30v22H33z" fill="#EAD9B2"/>
    <path d="M48 47c-5 2-7 7-6 11 1 3 3 5 6 5s5-2 6-5c1-4-1-9-6-11z" fill="{LEAF}" stroke="{INK}" stroke-width="2"/>
    <path d="M48 52v10" stroke="{INK}" stroke-width="2"/>
  </g>
</svg>
"""

LEAF_MARK = """<path d="M32 9C19 16 13 27 15 39c1.6 9 8.2 14 17 14s15.4-5 17-14c2-12-4-23-17-30z" fill="{leaf}"/>
  <g stroke="{vein}" stroke-linecap="round" fill="none">
    <path d="M32 16v42" stroke-width="3.2"/>
    <path d="M32 28l8-5M32 36l-8-5M32 44l9-5M32 50l-7-4" stroke-width="2.6"/>
  </g>"""

FAVICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="{INK}"/>
  {LEAF_MARK.format(leaf=LEAF, vein=PAPER)}
</svg>
"""

# Maskable icon: full-bleed background, leaf kept inside the central "safe zone".
MASKABLE_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" fill="{INK}"/>
  <g transform="translate(12.8 12.8) scale(0.6)">{LEAF_MARK.format(leaf=LEAF, vein=PAPER)}</g>
</svg>
"""

HERO_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 420">
  <circle cx="210" cy="210" r="200" fill="{BADGE}"/>
  <g stroke="{INK}" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M210 52c88 52 128 140 96 232-18 52-58 82-96 86-38-4-78-34-96-86-32-92 8-180 96-232z" fill="{LEAF}" stroke-width="5"/>
    <path d="M210 70v330" stroke-width="5"/>
    <path d="M210 140l44-32M210 140l-44-32M210 196l64-42M210 196l-64-42M210 254l70-40M210 254l-70-40M210 312l52-34M210 312l-52-34" stroke-width="4" opacity="0.75"/>
  </g>
  <g fill="#B4442E" opacity="0.85">
    <circle cx="160" cy="232" r="11"/><circle cx="176" cy="262" r="6"/><circle cx="252" cy="178" r="8"/><circle cx="246" cy="296" r="7"/>
  </g>
  <path d="M60 110V60h50M310 60h50v50M360 310v50h-50M110 360H60v-50" stroke="{INK}" stroke-width="7" stroke-linecap="round" fill="none"/>
  <path d="M74 214h272" stroke="{PAPER}" stroke-width="4" stroke-linecap="round" stroke-dasharray="2 12"/>
</svg>
"""

OG_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
@font-face {{ font-family: Fraunces; src: url("{fonts}/fraunces-var.woff2"); font-weight: 100 900; }}
@font-face {{ font-family: Inter; src: url("{fonts}/inter-var.woff2"); font-weight: 100 900; }}
html, body {{ margin: 0; width: 1200px; height: 630px; }}
body {{ display: flex; align-items: center; gap: 48px; padding: 0 80px; box-sizing: border-box;
       background: #F7F5EF; color: #1C2321; font-family: Inter; border-bottom: 18px solid #1F5135; }}
.text {{ flex: 1; }}
.logo {{ display: flex; align-items: center; gap: 18px; font: 700 44px Fraunces; color: #1F5135; }}
.logo img {{ width: 64px; height: 64px; }}
h1 {{ font: 600 76px/1.05 Fraunces; margin: 28px 0 24px; letter-spacing: -0.01em; }}
p {{ font-size: 30px; color: #5B6560; margin: 0; }}
.art {{ width: 400px; height: 400px; }}
</style></head><body>
<div class="text"><div class="logo"><img src="{img}/favicon.svg">LeafScan</div>
<h1>Leaf disease diagnosis in seconds</h1><p>Photograph one leaf. Get a diagnosis and care steps.</p></div>
<img class="art" src="{img}/hero-leaf.svg">
</body></html>"""


def write_svgs():
    (IMG / "crops").mkdir(parents=True, exist_ok=True)
    (IMG / "products").mkdir(parents=True, exist_ok=True)
    for name, parts in CROPS.items():
        (IMG / "crops" / f"{name}.svg").write_text(crop_svg(parts), encoding="utf-8")
    (IMG / "products" / "treatment.svg").write_text(TREATMENT_SVG, encoding="utf-8")
    (IMG / "products" / "fertiliser.svg").write_text(FERTILISER_SVG, encoding="utf-8")
    (IMG / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")
    (IMG / "hero-leaf.svg").write_text(HERO_SVG, encoding="utf-8")


def render_pngs():
    from playwright.sync_api import sync_playwright

    def svg_page(svg, size, padding=0, background="transparent"):
        inner = size - 2 * padding
        sized = svg.replace("<svg ", '<svg width="100%" height="100%" ', 1)
        return (f'<html><body style="margin:0;background:{background};width:{size}px;height:{size}px;'
                f'display:grid;place-items:center">'
                f'<div style="width:{inner}px;height:{inner}px">{sized}</div>'
                f'</body></html>')

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        jobs = [
            ("favicon-32.png", FAVICON_SVG, 32, 0, "transparent"),
            ("icon-192.png", FAVICON_SVG, 192, 0, "transparent"),
            ("icon-512.png", FAVICON_SVG, 512, 0, "transparent"),
            ("icon-maskable-512.png", MASKABLE_SVG, 512, 0, INK),
            ("apple-touch-icon.png", MASKABLE_SVG, 180, 0, INK),  # iOS adds its own rounded corners
        ]
        for filename, svg, size, padding, bg in jobs:
            page = browser.new_page(viewport={"width": size, "height": size})
            page.set_content(svg_page(svg, size, padding, bg))
            page.screenshot(path=str(IMG / filename), omit_background=(bg == "transparent"))
            page.close()

        og_file = ROOT / "scripts" / "_og.html"
        og_file.write_text(OG_HTML.format(fonts=(STATIC / "fonts").as_uri(), img=IMG.as_uri()), encoding="utf-8")
        page = browser.new_page(viewport={"width": 1200, "height": 630})
        page.goto(og_file.as_uri())
        page.wait_for_timeout(300)  # let the fonts load
        page.screenshot(path=str(IMG / "og-image.png"))
        browser.close()
        og_file.unlink()

    # favicon.ico for old browsers that request /favicon.ico directly
    Image.open(IMG / "icon-192.png").save(IMG / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])


if __name__ == "__main__":
    write_svgs()
    render_pngs()
    print("Artwork written to", IMG)
