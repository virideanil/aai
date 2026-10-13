"""Exercise each pinned tokenizer through its complete native text pipeline."""
import argparse
import base64
import importlib.metadata
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import digest_bytes, file_digest
from nanoai.syszero import read_kimi_pattern
from nanoai.tokenizer_sources import read_inventory


TEXTS = ["A careful observation changes the next decision.", "İstanbul’da ışık, düşünce ve ölçüm. ı I İ i",
         "  a\t b\n\nsecond\r\n ", "é é 🧬 👩🏽‍🔬 𐍈 雨日本語한글", "قال الباحث: مثال جديد। नमस्ते",
         "{\"x\":\"\\x41\\u0041\",\"n\":-1.25e-6}", "<|endoftext|> [UNK] ##a ▁b", "\x00\x01end"]


def native(spec, evidence):
    path, name = Path(spec["path"]), spec["name"]
    if spec["kind"] == "hf-tokenizer":
        from tokenizers import Tokenizer
        engine = Tokenizer.from_file(str(path))
        return (lambda text: engine.encode(text, add_special_tokens=False).ids,
                lambda ids: engine.decode(ids, skip_special_tokens=False).encode(),
                {"engine": "tokenizers.Tokenizer", "pipeline": "complete tokenizer.json; add_special_tokens=False"})
    if spec["kind"] == "sentencepiece":
        import sentencepiece
        engine = sentencepiece.SentencePieceProcessor(model_file=str(path))
        return (lambda text: engine.encode(text, out_type=int), lambda ids: engine.decode(ids).encode(),
                {"engine": "SentencePieceProcessor", "pipeline": "complete protobuf; IDs belong to this file, not an assumed HF wrapper"})
    if spec["kind"] == "wordpiece":
        from tokenizers import BertWordPieceTokenizer
        config_path = evidence / name / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        engine = BertWordPieceTokenizer(str(path), lowercase=config["do_lower_case"])
        return (lambda text: engine.encode(text, add_special_tokens=False).ids,
                lambda ids: engine.decode(ids, skip_special_tokens=False).encode(),
                {"engine": "BertWordPieceTokenizer", "configuration": config,
                 "configuration_identity": file_digest(config_path), "pipeline": "BERT normalization, pretokenization and WordPiece; no CLS/SEP"})
    if spec["kind"] == "tiktoken":
        import tiktoken
        ranks = {base64.b64decode(p, validate=True): int(i) for p, i in (line.split() for line in path.read_bytes().splitlines())}
        if name in ("cl100k_base", "o200k_base"):
            import tiktoken_ext.openai_public as public
            with patch.object(public, "load_tiktoken_bpe", return_value=ranks):
                config = getattr(public, name)()
            extra = {"rules_identity": file_digest(public.__file__)}
        else:
            root = path.parent if (path.parent / "tokenization_kimi.py").exists() else evidence / name
            pattern, reserved = read_kimi_pattern(root / "tokenization_kimi.py")
            source_config = json.loads((root / "tokenizer_config.json").read_text())
            declared = source_config["added_tokens_decoder"]
            controls = {declared.get(str(i), {"content": f"<|reserved_token_{i}|>"})["content"]: i
                        for i in range(len(ranks), len(ranks) + reserved)}
            config = {"name": name, "pat_str": pattern, "mergeable_ranks": ranks, "special_tokens": controls}
            extra = {"rules_identity": file_digest(root / "tokenization_kimi.py"),
                     "configuration_identity": file_digest(root / "tokenizer_config.json")}
        engine = tiktoken.Encoding(**config)
        return (lambda text: engine.encode(text, disallowed_special=()), engine.decode_bytes,
                {"engine": "tiktoken.Encoding", "pipeline": "source regex and rank table; reserved spellings treated as literal text", **extra})
    raise ValueError("Native adapter requires review")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = json.loads(args.sources.read_text())["sources"]
    result = {"schema": "nanoai.native-source-qualification.v1", "sources_identity": file_digest(args.sources),
              "libraries": {n: importlib.metadata.version(n) for n in ("tokenizers", "tiktoken", "sentencepiece", "protobuf")},
              "rows": [], "scope": "Native tokenizer text behavior and source ID qualification, without neural model execution"}
    for spec in sources:
        spec["path"] = str((args.sources.parent / spec["path"]).resolve())
        inventory = read_inventory(spec)
        encode, decode, rules = native(spec, args.evidence)
        observations = []
        for text in TEXTS:
            ids = encode(text)
            if not ids or any(type(i) is not int or i not in inventory.surfaces and i not in inventory.controls for i in ids):
                raise ValueError("Native ID outside source inventory: " + spec["name"])
            decoded = decode(ids)
            observations.append({"input_hex": text.encode().hex(), "native_ids": ids, "decoded_hex": decoded.hex(),
                "exact_roundtrip": decoded == text.encode(), "controls": [i for i in ids if i in inventory.controls],
                "surface_join_matches_native_decode": b"".join(inventory.surfaces.get(i, b"") for i in ids) == decoded})
        row = {"source": spec["name"], "payload_identity": inventory.metadata["payload_identity"], "rules": rules,
               "checks": len(observations), "exact_roundtrips": sum(o["exact_roundtrip"] for o in observations),
               "observations": observations}
        with (args.out / (spec["name"] + ".json")).open("x") as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2); stream.write("\n")
        result["rows"].append({k: v for k, v in row.items() if k != "observations"})
        print(json.dumps({"source": spec["name"], "checks": row["checks"], "exact": row["exact_roundtrips"]}), flush=True)
    result["checks"] = sum(r["checks"] for r in result["rows"])
    result["identity"] = digest_bytes(json.dumps(result, sort_keys=True, ensure_ascii=False).encode())
    with (args.out / "result.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")


if __name__ == "__main__":
    main()
