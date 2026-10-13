"""Build the isolated macOS CPU representation/projection library."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nanoai.core import file_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cxx", type=Path)
    args = p.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("This adapter requires Apple Accelerate; other backends need a qualified adapter")
    args.out.mkdir(parents=True, exist_ok=False)
    # Keep the clang++ driver name: resolving its symlink can select clang's C
    # link driver and omit the C++ runtime even though the source is .cpp.
    compiler = str(args.cxx.absolute()) if args.cxx else subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    source = ROOT / "native/harmony.cpp"
    target = args.out.resolve() / "libharmony.dylib"
    command = [compiler, "-isysroot", sdk, "-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
        "-Wno-deprecated-declarations", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC",
        str(source), "-framework", "Accelerate", "-pthread", "-o", str(target)]
    source_digest = file_digest(source)
    r = subprocess.run(command, capture_output=True, text=True)
    receipt = {"command": command, "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "source_digest": source_digest, "source_unchanged": file_digest(source) == source_digest,
        "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr,
        "binary_digest": file_digest(target) if r.returncode == 0 else None}
    (args.out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"returncode": r.returncode, "source_unchanged": receipt["source_unchanged"], "stderr": r.stderr[:1500]}))
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
