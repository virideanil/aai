"""Build the one native implementation; runtime performs no implicit compilation."""
import os
from pathlib import Path
import shlex
import subprocess
from setuptools import Extension, setup

root = os.environ.get("NANOAI_OPENSSL_ROOT")
if root:
    flags = ["-I" + str(Path(root) / "include"), "-L" + str(Path(root) / "lib"), "-lcrypto"]
else:
    try:
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "libcrypto"], text=True))
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("OpenSSL libcrypto is required. Install its development files and pkg-config, or set NANOAI_OPENSSL_ROOT to the existing OpenSSL prefix.") from error
setup(ext_modules=[Extension("nanoai_tokens._native", ["src/nanoai_tokens/native/python.cpp"],
    language="c++", extra_compile_args=["-std=c++20", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-ffp-contract=off"] + [x for x in flags if x.startswith("-I")],
    extra_link_args=[x for x in flags if not x.startswith("-I")])])
