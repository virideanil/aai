"""Build the native NMT1 consumer against an explicitly located OpenSSL."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import file_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--openssl-root", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Select a new output path for a versioned native build")
    source = Path(__file__).resolve().parents[1] / "native/master_tokenizer.cpp"
    if args.openssl_root:
        flags = ["-I" + str(args.openssl_root / "include"), "-L" + str(args.openssl_root / "lib"), "-lcrypto"]
    else:
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "libcrypto"], text=True))
    options = ["-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    options += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined"] if args.sanitize else ["-O3"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, *options, str(source), *flags, "-o", str(args.output)]
    subprocess.run(command, check=True)
    print(json.dumps({"source_identity": file_digest(source), "executable_identity": file_digest(args.output),
                      "command": command, "compiler": subprocess.check_output([args.compiler, "--version"], text=True).strip()}))


if __name__ == "__main__":
    main()
