from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from braillebyte.datasets import build_owned_repo_dataset


def test_build_deduplicates_and_rejects_secrets(tmp_path: Path):
    inventory = tmp_path / "repos.tsv"
    inventory.write_text("owner/repo\t1\tfalse\tfalse\tmain\n")
    checkout = tmp_path / "checkouts" / "owner__repo"
    checkout.mkdir(parents=True)
    (checkout / "agent.py").write_text("def plan():\n    return 'visual glyph agent'\n")
    (checkout / "copy.py").write_text("def plan():\n    return 'visual glyph agent'\n")
    (checkout / "config.py").write_text("api_key = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890'\n")
    output = tmp_path / "output"

    manifest = build_owned_repo_dataset(inventory, tmp_path / "checkouts", output, 500_000)

    assert manifest["total_records"] == 1
    assert manifest["skipped"]["duplicate_content"] == 1
    assert manifest["skipped"]["sensitive_content"] == 1
    records = []
    for split in ("train", "validation", "test"):
        with gzip.open(output / f"{split}.jsonl.gz", "rt") as handle:
            records.extend(json.loads(line) for line in handle)
    assert records[0]["metadata"]["source_repository"] == "owner/repo"
    assert "VISUAL" in records[0]["output"]["glyph_sequence"]
