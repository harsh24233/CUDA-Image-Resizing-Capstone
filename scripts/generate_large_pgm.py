#!/usr/bin/env python3
"""Generate a deterministic large P5 image for memory/throughput testing."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='data/large_test.pgm')
parser.add_argument('--width', type=int, default=10000)
parser.add_argument('--height', type=int, default=10000)
args = parser.parse_args()
if args.width <= 0 or args.height <= 0:
    parser.error('width and height must be positive')
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('wb') as out:
    out.write(f'P5\n{args.width} {args.height}\n255\n'.encode('ascii'))
    for y in range(args.height):
        out.write(bytes((x * 17 + y * 31 + (x ^ y)) & 0xFF for x in range(args.width)))
print(f'Created {path}: {args.width}x{args.height}, {args.width * args.height:,} pixel bytes')
