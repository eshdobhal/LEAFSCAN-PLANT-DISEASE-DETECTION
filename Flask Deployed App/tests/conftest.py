import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
TEST_IMAGES = APP_DIR.parent / "test_images"
sys.path.insert(0, str(APP_DIR))  # so `import app` works when pytest runs from anywhere

from app import create_app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    """One app for the whole test run: loading the model takes a second or two."""
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def model_loaded(app):
    return app.test_client().get("/healthz").get_json()["model_loaded"]
