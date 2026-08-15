from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TEXT_EXTENSIONS = {
    ".bash", ".c", ".cfg", ".cpp", ".css", ".csv", ".go", ".h", ".hpp",
    ".html", ".ini", ".ipynb", ".java", ".js", ".json", ".jsonl", ".jsx",
    ".kt", ".md", ".php", ".ps1", ".py", ".rb", ".rs", ".sh", ".sql",
    ".svg", ".toml", ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
SKIP_PARTS = {".git", ".hg", ".svn", ".venv", "__pycache__", "build", "dist", "node_modules", "site-packages", "target", "vendor"}
SENSITIVE_NAMES = re.compile(r"(^|/)(\.env($|\.)|id_(rsa|ed25519)|.*credentials?.*|.*secrets?.*|.*tokens?.*|hosts\.yml|cookies?\.txt)", re.IGNORECASE)
SENSITIVE_CONTENT = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password)\b\s*[:=]\s*['\"](?!example|placeholder|changeme|test)[^'\"\s]{12,}['\"]"),
]
LAYER_TERMS = {
    "AGENT": ("agent", "planner", "orchestrat", "tool_call"),
    "MODEL": ("model", "tensor", "inference", "embedding", "lora"),
    "DATA": ("dataset", "corpus", "jsonl", "loader", "ingest"),
    "VISUAL": ("vision", "visual", "image", "glyph", "canvas", "render"),
    "RUNTIME": ("runtime", "execute", "server", "cli", "pipeline"),
    "SECURITY": ("security", "hash", "sha256", "encrypt", "verify"),
    "EVAL": ("test", "benchmark", "score", "eval", "kaggle", "arc-agi"),
    "DOCS": ("readme", "documentation", ".md"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def layers_for(path: str, content: str) -> list[str]:
    sample = (path + "\n" + content[:20_000]).lower()
    layers = [name for name, terms in LAYER_TERMS.items() if any(term in sample for term in terms)]
    return layers or ["CODE"]


def glyph_sequence(layers: list[str], digest: str) -> list[str]:
    return ["OPEN", *layers[:6], f"HASH_{digest[:8].upper()}", "VALIDATE", "CLOSE"]


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if SENSITIVE_NAMES.search(relative):
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS or path.name.upper().startswith(("README", "LICENSE")):
            yield path


def read_inventory(path: Path) -> list[dict[str, Any]]:
    repos = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        owner_repo, size_kb, private, fork, branch = line.split("\t")
        repos.append({
            "name_with_owner": owner_repo,
            "size_kb": int(size_kb or 0),
            "private": private == "true",
            "fork": fork == "true",
            "default_branch": branch,
        })
    return repos


def build_owned_repo_dataset(inventory_path: Path, checkout_root: Path, output: Path, max_file_bytes: int) -> dict[str, Any]:
    repos = read_inventory(inventory_path)
    output.mkdir(parents=True, exist_ok=True)
    handles = {split: gzip.open(output / f"{split}.jsonl.gz", "wt", encoding="utf-8") for split in ("train", "validation", "test")}
    seen: set[str] = set()
    counts = Counter()
    repo_counts = Counter()
    skipped = Counter()
    try:
        for repo in repos:
            owner_repo = repo["name_with_owner"]
            root = checkout_root / owner_repo.replace("/", "__")
            if not root.exists():
                root = checkout_root / owner_repo.replace("/", "_")
            if not root.exists():
                skipped["repository_unavailable"] += 1
                continue
            for path in iter_files(root):
                relative = path.relative_to(root).as_posix()
                if path.stat().st_size > max_file_bytes:
                    skipped["file_too_large"] += 1
                    continue
                data = path.read_bytes()
                if b"\0" in data:
                    skipped["binary"] += 1
                    continue
                content = data.decode("utf-8", errors="replace")
                if any(pattern.search(content) for pattern in SENSITIVE_CONTENT):
                    skipped["sensitive_content"] += 1
                    continue
                digest = sha256(content.encode("utf-8"))
                if digest in seen:
                    skipped["duplicate_content"] += 1
                    continue
                seen.add(digest)
                layers = layers_for(relative, content)
                bucket = int(digest[:8], 16) % 100
                split = "train" if bucket < 90 else "validation" if bucket < 95 else "test"
                record = {
                    "id": f"repo-{digest[:24]}",
                    "version": "1.0.0",
                    "task_type": "repository_source_to_glyphmatic_vil",
                    "instruction": "Encode owned repository source into a provenance-preserving GlyphMatics VIL representation.",
                    "input": {"source_text": content},
                    "output": {
                        "canonical_construct": ">".join(glyph_sequence(layers, digest)),
                        "glyph_sequence": glyph_sequence(layers, digest),
                        "layers": layers,
                        "compressed_form": "|".join(layers + [digest[:16]]),
                        "source_sha256": digest,
                    },
                    "metadata": {
                        "source_repository": owner_repo,
                        "source_path": relative,
                        "source_private": repo["private"],
                        "source_fork": repo["fork"],
                        "default_branch": repo["default_branch"],
                        "author_namespace": owner_repo.split("/", 1)[0],
                        "organization": "918 Technologies",
                        "redistribution": "private" if repo["private"] else "owner_asserted",
                    },
                    "record_hash": sha256(f"{owner_repo}\0{relative}\0{digest}".encode()),
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                repo_counts[owner_repo] += 1
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "format": "GlyphMatics Owned Repository VIL Dataset",
        "version": "1.0.0",
        "repositories_declared": len(repos),
        "repositories_ingested": len(repo_counts),
        "records": dict(counts),
        "total_records": sum(counts.values()),
        "repository_records": dict(sorted(repo_counts.items())),
        "skipped": dict(skipped),
        "privacy": "Contains private-source records; keep private unless explicitly reviewed.",
        "inventory": repos,
    }
    manifest["manifest_sha256"] = sha256(json.dumps(manifest, sort_keys=True).encode())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_image_only_dataset(source: Path, output: Path, *, lines_per_page: int = 64, columns: int = 108, font_size: int = 14, resume: bool = False, max_records: int = 0) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    font_candidates = (
        "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/system/fonts/RobotoMono-Regular.ttf",
        "/system/fonts/DroidSansMono.ttf",
    )

    def choose_font(size: int):
        for candidate in font_candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    def records(path: Path) -> Iterable[dict[str, Any]]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def visible_lines(record: dict[str, Any], width: int) -> list[str]:
        output = record["output"]
        metadata = record["metadata"]
        header = [
            "pIMAGEVIL1",
            f"ID {record['id']}",
            f"GLYPHS {'>'.join(output['glyph_sequence'])}",
            f"SOURCE {metadata['source_repository']}::{metadata['source_path']}",
            f"SHA256 {output['source_sha256']}",
            f"PRIVACY {'PRIVATE' if metadata['source_private'] else 'PUBLIC'}",
            "BEGIN_LITERAL_SOURCE",
        ]
        source = record["input"]["source_text"].replace("\t", "    ")
        body: list[str] = []
        for line in source.splitlines() or [""]:
            wrapped = textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False, break_long_words=True, break_on_hyphens=False)
            body.extend(wrapped or [""])
        return header + body + ["END_LITERAL_SOURCE", "pIMAGEVIL_CLOSE"]

    def render_page(lines: list[str], output: Path) -> dict[str, Any]:
        font = choose_font(font_size)
        line_height = max(font_size + 5, 16)
        margin = 24
        width = max(768, margin * 2 + int(columns * font_size * 0.64))
        height = margin * 2 + line_height * len(lines)
        image = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(image)
        y = margin
        for line in lines:
            draw.text((margin, y), line, font=font, fill=0)
            y += line_height
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG", compress_level=3)
        data = output.read_bytes()
        return {"path": output.as_posix(), "sha256": sha256(data), "bytes": len(data), "width": width, "height": height}

    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    completed: set[str] = set()
    if resume and index_path.exists():
        completed = {entry["id"] for entry in valid_index_entries(index_path)}
    counts = Counter()
    processed_now = 0
    with index_path.open("a" if resume else "w", encoding="utf-8", buffering=1) as index:
        for split in ("train", "validation", "test"):
            for record in records(source / f"{split}.jsonl.gz"):
                if record["id"] in completed:
                    continue
                if max_records and processed_now >= max_records:
                    break
                lines = visible_lines(record, columns)
                pages = []
                for page_number, start in enumerate(range(0, len(lines), lines_per_page)):
                    relative = Path(split) / record["id"] / f"page-{page_number:04d}.png"
                    info = render_page(lines[start:start + lines_per_page], output / relative)
                    info["path"] = relative.as_posix()
                    pages.append(info)
                entry = {
                    "id": record["id"],
                    "split": split,
                    "images": pages,
                    "source_repository": record["metadata"]["source_repository"],
                    "source_path": record["metadata"]["source_path"],
                    "source_private": record["metadata"]["source_private"],
                    "source_sha256": record["output"]["source_sha256"],
                    "record_hash": record["record_hash"],
                }
                index.write(json.dumps(entry, ensure_ascii=False) + "\n")
                counts["records"] += 1
                counts[split] += 1
                counts["images"] += len(pages)
                processed_now += 1
            if max_records and processed_now >= max_records:
                break
    entries = valid_index_entries(index_path)
    repaired_index = index_path.with_suffix(".jsonl.tmp")
    repaired_index.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")
    repaired_index.replace(index_path)
    manifest = {
        "format": "GlyphMatics Literal Image-Only VIL Dataset",
        "version": "1.0.0",
        "transport": "visible grayscale PNG text; no hidden payload or byte-to-glyph wrapper",
        "model_input": "image_only",
        "records": len(entries),
        "images": sum(len(entry["images"]) for entry in entries),
        "splits": {name: Counter(entry["split"] for entry in entries)[name] for name in ("train", "validation", "test")},
        "private_images": sum(len(entry["images"]) for entry in entries if entry["source_private"]),
        "image_bytes": sum(page["bytes"] for entry in entries for page in entry["images"]),
        "lines_per_page": lines_per_page,
        "columns": columns,
        "font_size": font_size,
        "source_manifest_sha256": json.loads((source / "manifest.json").read_text())["manifest_sha256"],
        "privacy": "Contains rasterized private-source material; keep private unless reviewed.",
        "complete": len(entries) == json.loads((source / "manifest.json").read_text())["total_records"],
    }
    manifest["index_sha256"] = sha256(index_path.read_bytes())
    manifest["manifest_sha256"] = sha256(json.dumps(manifest, sort_keys=True).encode())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def valid_index_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("id") and entry.get("images"):
            entries.append(entry)
    return entries
