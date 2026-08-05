"""Create complete-system BrailleByte sentences and train a lossless compressor."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from braillebyte.compression import BrailleByteCompressor

START, END, CLAUSE = 1, 2, 3
AGENT, ACTION, PATIENT, PROPERTY, LOCATION, TIME = 32, 33, 34, 36, 37, 38
EXT_CONCEPT = 255


def varuint(value: int) -> list[int]:
    result: list[int] = []
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return result


def concept(concept_id: int) -> list[int]:
    return [EXT_CONCEPT, *varuint(concept_id)]


def build_system_corpus() -> list[dict[str, object]]:
    system_types = [('edge analytics', 1001), ('local assistant', 1002), ('resilient service', 1003)]
    deployments = [('offline', 1101), ('hybrid', 1102), ('networked', 1103)]
    profiles = [('safe', 1201), ('realtime', 1202), ('auditable', 1203)]
    components = [('controller', 2001), ('network', 2002), ('storage', 2003), ('security', 2004), ('power', 2005), ('telemetry', 2006)]
    records: list[dict[str, object]] = []
    for system_name, system_id in system_types:
        for deployment_name, deployment_id in deployments:
            for profile_name, profile_id in profiles:
                stream = [START, CLAUSE, AGENT, *concept(system_id), PROPERTY, *concept(deployment_id), PROPERTY, *concept(profile_id)]
                for component_name, component_id in components:
                    stream.extend((CLAUSE, PATIENT, *concept(component_id), ACTION, *concept(3001), PROPERTY, *concept(3100 + component_id % 10), TIME, 38))
                stream.extend((CLAUSE, LOCATION, *concept(4001), END))
                records.append({
                    'source': f'{profile_name} {deployment_name} {system_name} with controller, network, storage, security, power, and telemetry.',
                    'system': {'type': system_name, 'deployment': deployment_name, 'profile': profile_name, 'components': [name for name, _ in components]},
                    'bytes': stream,
                })
    return records


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    corpus = build_system_corpus()
    corpus_path = data_dir / 'system_sentence_corpus.jsonl'
    corpus_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in corpus), encoding='utf-8')
    split = len(corpus) * 80 // 100
    model = BrailleByteCompressor.train((row['bytes'] for row in corpus[:split]), min_frequency=4)
    held_out = [row['bytes'] for row in corpus[split:]]
    compact = [model.compress(stream) for stream in held_out]
    if any(model.decompress(encoded) != tuple(original) for original, encoded in zip(held_out, compact)):
        raise RuntimeError('compression round-trip failed')
    model_path = data_dir / 'system_compression_model.json'
    model_path.write_text(json.dumps(model.to_dict(), indent=2) + '\n', encoding='utf-8')
    original_bytes = sum(len(stream) for stream in held_out)
    compressed_bytes = sum(len(stream) for stream in compact)
    report = {
        'training_sentences': split,
        'held_out_complete_system_sentences': len(held_out),
        'learned_phrases': len(model.phrases),
        'held_out_canonical_bytes': original_bytes,
        'held_out_compressed_bytes': compressed_bytes,
        'reduction_percent': round(100 * (1 - compressed_bytes / original_bytes), 2),
        'exact_round_trip': True,
    }
    (data_dir / 'system_compression_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
