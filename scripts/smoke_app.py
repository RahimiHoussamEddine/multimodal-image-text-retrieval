"""Exercise text search and the top-k control through Streamlit AppTest."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=180).run()
    if app.exception:
        raise AssertionError("App startup failed: " + str(app.exception[0].message))
    app.text_input(key="text_query").set_value("a person riding a bicycle")
    app.slider(key="top_k").set_value(3)
    app.button(key="search_text").click().run(timeout=180)
    if app.exception:
        raise AssertionError("Text-query interaction failed: " + str(app.exception[0].message))
    page_text = "\n".join(element.value for element in app.markdown)
    if "Search results for" not in page_text:
        raise AssertionError("Text search did not render result content")

    root = app_path.parent
    corpus = json.loads((root / "data/processed/corpus.json").read_text(encoding="utf-8"))
    image_path = root / "data/raw/images" / corpus["images"][0]["file_name"]
    uploader = app.file_uploader(key="image_query")
    uploader.set_value((image_path.name, image_path.read_bytes(), "image/jpeg"))
    app.button(key="search_image").click().run(timeout=180)
    if app.exception:
        raise AssertionError("Image-query interaction failed: " + str(app.exception[0].message))
    image_page_text = "\n".join(element.value for element in app.markdown)
    if "Image-to-image results" not in image_page_text:
        raise AssertionError("Image query did not render similar-image results")
    print(
        "Streamlit startup, text search, top-k control, uploaded-image search, "
        "and result rendering passed"
    )


if __name__ == "__main__":
    main()
