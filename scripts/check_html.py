"""Lightweight HTML check of every rendered page: unclosed/mismatched tags, duplicate ids,
images without alt, and inline style attributes (blocked by our CSP).

    .venv/Scripts/python scripts/check_html.py
"""
import io
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Flask Deployed App"))
from app import create_app  # noqa: E402

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr", "use"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.ids, self.problems = [], set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            if attrs["id"] in self.ids:
                self.problems.append(f"duplicate id {attrs['id']!r}")
            self.ids.add(attrs["id"])
        if tag == "img" and "alt" not in attrs:
            self.problems.append(f"img without alt: {attrs.get('src', '')[:60]}")
        if "style" in attrs:
            self.problems.append(f"inline style on <{tag}>")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"unexpected </{tag}>, open: {self.stack[-3:]}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def main():
    app = create_app()
    client = app.test_client()
    sample = (ROOT / "test_images" / "Apple_scab.JPG").read_bytes()
    pages = {url: client.get(url) for url in ["/", "/diagnose", "/treatments", "/about", "/missing"]}
    pages["result"] = client.post("/diagnose", data={"image": (io.BytesIO(sample), "a.jpg")},
                                  content_type="multipart/form-data")
    total = 0
    for name, response in pages.items():
        checker = Checker()
        checker.feed(response.get_data(as_text=True))
        if checker.stack:
            checker.problems.append(f"unclosed tags: {checker.stack}")
        total += len(checker.problems)
        print(f"{name:12} {'OK' if not checker.problems else checker.problems}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
