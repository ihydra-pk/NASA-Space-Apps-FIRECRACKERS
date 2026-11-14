"""Command-line entry point: ``python -m src.chunking``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import ChunkingConfig, run_chunking


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk normalized NASA document records.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--max-chunk-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=120)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_chunking(
        ChunkingConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_chunk_tokens=args.max_chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
    )
    print(
        f"Processed {result.input_records} input record(s) into "
        f"{len(result.chunks)} chunk(s); overlap={args.overlap_tokens} estimated tokens."
    )
    print(f"Records producing zero chunks: {len(result.zero_chunk_records)}")
    if result.errors:
        print(f"{len(result.errors)} input file(s) could not be processed:")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
