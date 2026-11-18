"""Loads the two CSV files once and turns each row into a tidy `Condition` object.

Row N of each CSV describes class N of the model (same order as CNN.idx_to_classes),
so the list index is the class index.
"""
import csv
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from CNN import idx_to_classes

# Display fixes. The CSV keys stay untouched; only the text people read changes.
CROP_NAMES = {"Pepper bell": "Bell pepper"}
CONDITION_NAMES = {
    "Haunglongbing | Citrus Greening": "Huanglongbing (citrus greening)",
    "Cedar rust": "Cedar apple rust",
}
# Every crop has an illustration in static/img/crops/<slug>.svg
CROP_SLUGS = {"Bell pepper": "pepper"}

BACKGROUND_KEY = "Background_without_leaves"


@dataclass(frozen=True)
class Condition:
    index: int            # model class index (0..38)
    key: str              # raw class name, e.g. "Apple___Apple_scab"
    crop: str             # "Apple"
    crop_slug: str        # "apple" (used for image file names and filters)
    condition: str        # "Apple scab"
    is_healthy: bool
    is_background: bool   # the "no leaf in the photo" class
    description: list     # list of sentences/paragraphs
    steps: list           # treatment steps (or benefits, for healthy classes)
    ref_image: str        # external reference photo URL (may break; UI hides it then)
    product_name: str
    product_link: str

    @property
    def label(self) -> str:
        """Human-readable name used on every page, e.g. "Tomato · Late blight"."""
        if self.is_background:
            return self.condition
        return f"{self.crop} · {self.condition}"

    @property
    def product_kind(self) -> str:
        """Healthy plants get a fertiliser suggestion, diseased ones a treatment."""
        return "fertiliser" if self.is_healthy else "treatment"


def _split_lines(text: str) -> list:
    """CSV cells contain several sentences separated by newlines -> list of strings.
    Also repairs a common typo in the data: a missing space after a full stop."""
    text = re.sub(r"\.(?=[A-Z])", ". ", text or "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _display_names(disease_name: str):
    """'Grape : Esca | Black Measles' -> ('Grape', 'Esca (black measles)')."""
    crop, _, condition = (part.strip() for part in disease_name.partition(":"))
    crop = CROP_NAMES.get(crop, crop)
    condition = CONDITION_NAMES.get(condition, condition)
    if "|" in condition:  # 'Main | Alias' -> 'Main (alias)'
        main, alias = (p.strip() for p in condition.split("|", 1))
        condition = f"{main} ({alias.lower()})"
    # Sentence case: "Leaf Mold" -> "Leaf mold", but keep words like "Two-Spotted" readable.
    condition = condition[:1].upper() + condition[1:].lower() if condition else condition
    return crop, condition


def _read_csv(path):
    # The files were saved from Excel on Windows, hence cp1252. Python turns them into normal str.
    with open(path, encoding="cp1252", newline="") as f:
        return list(csv.DictReader(f))


def load_catalog(disease_csv, supplement_csv) -> list:
    diseases = _read_csv(disease_csv)
    supplements = _read_csv(supplement_csv)
    if not (len(diseases) == len(supplements) == len(idx_to_classes)):
        raise ValueError("CSV row count does not match the model's class count")

    catalog = []
    for index, (d, s) in enumerate(zip(diseases, supplements)):
        key = idx_to_classes[index]
        if s["disease_name"] != key:  # guards against the CSVs being re-ordered
            raise ValueError(f"supplement_info.csv row {index} is {s['disease_name']!r}, expected {key!r}")
        is_background = key == BACKGROUND_KEY
        crop, condition = _display_names(d["disease_name"])
        if is_background:
            crop, condition = "", "No leaf detected"
        catalog.append(Condition(
            index=index,
            key=key,
            crop=crop,
            crop_slug=CROP_SLUGS.get(crop, crop.lower()),
            condition=condition,
            is_healthy=key.endswith("healthy"),
            is_background=is_background,
            description=_split_lines(d["description"]),
            steps=_split_lines(d["Possible Steps"]),
            # http:// images would be blocked as "mixed content" on an https site.
            ref_image=d["image_url"].strip().replace("http://", "https://", 1),
            product_name=" ".join((s["supplement name"] or "").split()),
            product_link=(s["buy link"] or "").strip(),
        ))
    return catalog


def crop_summary(catalog) -> list:
    """One entry per crop, in alphabetical order, with how many diseases the model knows for it."""
    crops = {}
    for c in catalog:
        if c.is_background:
            continue
        entry = crops.setdefault(c.crop, {"name": c.crop, "slug": c.crop_slug, "diseases": 0})
        if not c.is_healthy:
            entry["diseases"] += 1
    return sorted(crops.values(), key=lambda e: e["name"])


def image_hosts(catalog) -> set:
    """Hosts of the external reference photos, needed for the Content-Security-Policy."""
    return {urlparse(c.ref_image).netloc for c in catalog if c.ref_image.startswith("https://")}
