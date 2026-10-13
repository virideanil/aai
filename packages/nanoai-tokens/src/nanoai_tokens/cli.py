"""Local byte-file interface. Outputs are explicitly named and never overwritten."""
import argparse
import json
from pathlib import Path
import sys
from . import load


def main():
    parser = argparse.ArgumentParser(description="Exact-byte native tokenization")
    parser.add_argument("--model", type=Path, help="NMT1 model; defaults to the bundled frozen version")
    commands = parser.add_subparsers(dest="command", required=True)
    enc = commands.add_parser("encode")
    enc.add_argument("source", type=Path)
    enc.add_argument("--out", type=Path, required=True)
    enc.add_argument("--objective", choices=("likelihood", "minimum"), default="likelihood")
    enc.add_argument("--boundary", type=int, action="append", default=[])
    dec = commands.add_parser("decode")
    dec.add_argument("record", type=Path)
    dec.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect")
    args = parser.parse_args()
    try:
        tokenizer = load(args.model)
        if args.command == "encode":
            record = tokenizer.encode(args.source.read_bytes(), objective=args.objective, boundaries=args.boundary)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
        elif args.command == "decode":
            raw = tokenizer.decode(json.loads(args.record.read_text(encoding="utf-8")))
            with args.out.open("xb") as stream:
                stream.write(raw)
        else:
            print(json.dumps({"backend":"C++20", "tokenizer_identity":tokenizer.identity,
                              "pieces":len(tokenizer.pieces), "maximum_input_bytes":131072}, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
