"""Evidence gate: byte-exact round trips and standard lossless baselines."""
from __future__ import annotations
import argparse, bz2, gzip, json, lzma, sys, time
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from braillebyte import BrailleByteCodec
def measure(name, raw, encode, decode):
 start=time.perf_counter(); packed=encode(raw); enc=time.perf_counter()-start
 start=time.perf_counter(); restored=decode(packed); dec=time.perf_counter()-start
 return {'name':name,'bytes':len(packed),'ratio':round(len(raw)/max(1,len(packed)),4),'encode_seconds':round(enc,6),'decode_seconds':round(dec,6),'sha256':sha256(restored).hexdigest(),'exact':restored==raw}
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',default='data/benchmark_report.json');a=p.parse_args()
 raw=Path(a.input).read_bytes(); rows=[measure('gzip',raw,gzip.compress,gzip.decompress),measure('bz2',raw,bz2.compress,bz2.decompress),measure('lzma',raw,lzma.compress,lzma.decompress)]
 report={'input':a.input,'input_bytes':len(raw),'input_sha256':sha256(raw).hexdigest(),'results':rows,'sota_claim':False,'rule':'No SOTA claim is permitted until an independently reproducible benchmark beats a named baseline.'}
 if not all(row['exact'] for row in rows): raise RuntimeError('lossless baseline failed')
 Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
