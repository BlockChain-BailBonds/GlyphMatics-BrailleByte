from __future__ import annotations

import json
from pathlib import Path
import sys
import gzip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from braillebyte.datasets import build_image_only_dataset


def test_image_only_dataset_has_no_source_text_in_index(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    record = {
        "id": "repo-abc",
        "input": {"source_text": "def glyph():\n    return 'visible'"},
        "output": {"glyph_sequence": ["OPEN", "VISUAL", "CLOSE"], "source_sha256": "a" * 64},
        "metadata": {"source_repository": "owner/repo", "source_path": "a.py", "source_private": True},
        "record_hash": "b" * 64,
    }
    for split in ("train", "validation", "test"):
        with gzip.open(source / f"{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
            if split == "train":
                handle.write(json.dumps(record) + "\n")
    (source / "manifest.json").write_text(json.dumps({"manifest_sha256": "c" * 64, "total_records": 1}))
    output = tmp_path / "images"

    manifest = build_image_only_dataset(source, output, lines_per_page=20, columns=60, font_size=12)

    assert manifest["records"] == 1
    assert manifest["images"] == 1
    index = (output / "index.jsonl").read_text()
    assert "def glyph" not in index
    entry = json.loads(index)
    assert entry["source_private"] is True
    assert (output / entry["images"][0]["path"]).read_bytes().startswith(b"\x89PNG")
