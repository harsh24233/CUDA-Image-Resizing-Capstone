#!/usr/bin/env python3
"""Validate P5/8-bit PGM dimensions and pixel payload length (no CUDA required)."""
import sys
from pathlib import Path


def read_token(data: bytes, pos: int):
    n = len(data)
    while pos < n:
        if data[pos] in b" \t\r\n":
            pos += 1
        elif data[pos] == ord("#"):
            newline = data.find(b"\n", pos)
            if newline < 0:
                return None, n
            pos = newline + 1
        else:
            break
    start = pos
    while pos < n and data[pos] not in b" \t\r\n#":
        pos += 1
    return data[start:pos], pos


def validate(path: Path):
    data = path.read_bytes()
    pos = 0
    tokens = []
    for _ in range(4):
        token, pos = read_token(data, pos)
        if token is None:
            raise ValueError("incomplete PGM header")
        tokens.append(token)
    magic, width, height, maxval = tokens
    if magic != b"P5" or int(width) <= 0 or int(height) <= 0 or maxval != b"255":
        raise ValueError("expected P5 PGM with positive dimensions and maxval 255")
    # Consume the single whitespace separator following maxval.
    if pos >= len(data) or data[pos] not in b" \t\r\n":
        raise ValueError("missing header/pixel separator")
    pos += 1
    expected = int(width) * int(height)
    actual = len(data) - pos
    if actual != expected:
        raise ValueError(f"expected {expected} pixels, found {actual}")
    return int(width), int(height), actual


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: validate_pgm.py IMAGE.pgm [IMAGE2.pgm ...]")
    for arg in sys.argv[1:]:
        w, h, count = validate(Path(arg))
        print(f"OK: {arg}: {w}x{h}, {count} pixel bytes")
