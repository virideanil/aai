# The master tokenizer

The master tokenizer gives a new nano'ai consumer one stable way to turn bytes
into model rows and recover the original bytes. It keeps the source dictionary's
IDs alongside a compact active vocabulary. Python and C++ implement the same
finite segmentation law.

Version 1 is frozen in [master.nmt](../data/master-tokenizer-v1/master.nmt).
Its [manifest](../data/master-tokenizer-v1/manifest.json) identifies the source
payloads, selection policy, implementation and measured qualification. This
version is ready for the declared tokenizer/data experiments. Useful language
and semantic learning require their own trained consumer.

| Property | Frozen version |
|---|---:|
| Source payloads | 22 |
| Distinct source byte-piece sets | 19 |
| Unique pieces in the cold inventory | 909,421 |
| Active literal rows | 24,664 |
| Model rows, including BOS/EOS/PAD | 24,667 |
| Artifact size | 426,628 bytes |

Qwen3.8 and actual Kimi K3 are included. Source names, native IDs, contextual roles,
normalization behavior and individual license declarations remain attached to
their source versions. Repeated byte sets are counted once when comparing source
families. A source witness establishes membership in that tokenizer's vocabulary.
The [source notes](../data/master-tokenizer-v1/SOURCE-NOTES.md) list every pinned
payload and its recorded license declaration, with captured attribution files
preserved unchanged.

## Encode and recover bytes

Run this in the installed source checkout described by the [README](../README.md):

```python
from nanoai.master_tokenizer import MasterTokenizer

codec = MasterTokenizer.load("data/master-tokenizer-v1/master.nmt")
raw = "İstanbul, 雨 and 🧬".encode("utf-8") + b"\x00\xff"
record = codec.encode(raw)
assert codec.decode(record) == raw
```

The record carries the tokenizer identity, dictionary identity, original byte
digest and a sequence of spans. Each span names a common `id`, an `active_id`,
and half-open `start`/`end` byte offsets. The common ID identifies the piece in
System Zero's dictionary. The active ID indexes the new consumer's embedding
and output rows.

All 256 single-byte pieces are present. A piece can contain part of a UTF-8
character; completed bytes are decoded for human display. Input is preserved
without Unicode normalization. Existing pretrained models continue to receive
the complete original text through their own tokenizer and template, including
their source-specific normalization and control handling.

For a particular deployment or recorded run, supply the expected
`tokenizer_identity` from its reviewed manifest to `MasterTokenizer.load`.
The file's footer checks internal integrity; the separately supplied expected
identity binds the reader to the selected version.

## One probability law

Every active nonempty piece has `q(piece) > 0`; emission probabilities sum to one
within the format's floating-point validation tolerance. At each step, the process
stops with probability `p_end`, or emits a piece with probability
`(1 - p_end) * q(piece)`. `p_end` is strictly between zero and one.

For a complete byte string, sum the probabilities of all segmentations and their
terminal stop. The empty string has probability `p_end`. Forward probability,
Viterbi decoding, conditional samples and expected counts all use this law.
An exactly equal floating-point Viterbi score prefers the shorter final piece,
recursively. This deterministic tie rule is shared by the native implementation.

The formulation defines a finite-piece distribution over arbitrary finite byte
strings. Likelihood measures the selected lexical law. It supplies no probability
that an interpretation, factual claim or action is correct. Those errors require
their own task evidence and declared assumptions.

## How the active set was selected

The first calibration fitted 94 authored English/Turkish lines and selected on
10 validation lines. Its 4,096-row vocabulary reconstructed bytes, but a fresh
20-row diagnostic showed that six non-Latin script samples used byte fallbacks
throughout. That narrow result is retained as `frozen-v1` in the parent audit.

The revised policy protects every complete Unicode scalar attested in the source
inventory. The calibrated vocabulary keeps 98% of emission mass; 20,568 added
scalar rows share an explicit 2% reserve. All byte fallbacks remain. This prior
covers source-attested characters without claiming that their frequencies were
learned from the small calibration corpus.

Six recipes compared 1,024, 2,048 and 4,096 calibrated rows, each with a source-family
selection bonus of either 0 or 0.25. Selection chose the smallest active vocabulary
within 1% of the best validation nats per byte, then lower validation loss, then
lower family bonus. The selected recipe used 4,096 calibrated rows and no bonus.
Adding the protected alphabet gives the frozen 24,664 rows.

The selected version measured **1.995597 validation nats/byte**. A fresh, separately
frozen 16-row confirmation measured **4.445170 nats/byte**. This is a small authored
lexical diagnostic. Its size, domain differences and declared alphabet prior
remain material limits on any broader tokenizer-quality claim.

## Prepare causal model inputs

```python
from nanoai.master_data import causal_batch, validate_batch

batch = causal_batch(codec, [b"one", b"two\x00\xff"])
assert validate_batch(codec, batch) == [b"one", b"two\x00\xff"]
```

For `N` active pieces, the model contract reserves `N` for BOS, `N+1` for EOS and
`N+2` for PAD. These controls have no literal byte surface. Input rows contain
BOS followed by piece IDs; labels contain those pieces followed by EOS. Padding
labels use `-100`. A batch exceeding its declared budget is rejected with the
original data intact.

Saved batches bind the tokenizer and vocabulary identities and revalidate the
shifted IDs, labels, masks and byte restoration. A fixed BF16 embedding/output
fixture accepted `[5, 56]` inputs and produced `[5, 56, 24667]` logits with a finite
loss and unchanged parameter bytes. It checks indexing and data compatibility;
it is an untrained fixture.

## Build the native consumer

Use a C++20 compiler and OpenSSL libcrypto with SHAKE256 support. The helper uses
`pkg-config` by default, or accepts `--openssl-root` for an explicit installation:

```sh
python tools/build_master_tokenizer.py --output .local/master-tokenizer
python tools/build_master_tokenizer.py --output .local/master-tokenizer-asan --sanitize
```

Both commands require a new output path. `--compiler` selects the compiler.
The helper emits the command, compiler version and source/executable identities.
On the qualified macOS host, supplying the installed OpenSSL 3 prefix completed
the strict-warning build and matched all 127 saved full-artifact Python probes.

The executable takes the artifact path and an optional expected tokenizer
identity. Each stdin line is a hexadecimal byte string. A blank line represents
empty bytes; embedded input newlines are encoded as `0a`. Each stdout line is a
complete JSON record with restored `decoded_hex` and the encode timing:

```sh
.local/master-tokenizer data/master-tokenizer-v1/master.nmt <<'HEX'
68656c6c6f00ff
HEX
```

Embedding code can define `NANOAI_MASTER_LIBRARY_ONLY` before including the
translation unit. `tests/native_master_api.cpp` exercises its owned data lifetime,
record validation and malformed-input rejection. `NANOAI_MASTER_NATIVE` selects
an existing executable for Python/native differential tests:

```sh
NANOAI_MASTER_NATIVE=.local/master-tokenizer PYTHONPATH=src \
  python -m unittest discover -s tests -p test_master_tokenizer.py -v
```

## Format and bounds

`NMT1` is little-endian: magic bytes; a `uint32` row count; binary64 stop probability;
two 64-byte identities; and ordered `(uint32 common_id, uint16 length, binary64 q,
piece_bytes)` records. A 64-byte SHAKE256 footer hashes the preceding body.
The tokenizer identity hashes that body; the whole-file identity also includes
the footer. They are deliberately distinct fields.

The implementation accepts at most 131,072 active rows, 128 bytes per piece,
131,072 input bytes and 16,777,216 candidate-length checks per operation.
Oversized, malformed, duplicate or incompatible records fail explicitly.
A new piece set, ID mapping or probability law creates a new artifact version
and requires a compatible model consumer.

## Qualification and cost

An independent exhaustive oracle enumerated all segmentations of every string
over `a` and `b` through length six. A 16,000-sample check exercised conditional
sampling against that law. Strict C++ warnings, ASan/UBSan, malformed-container
cases, API ownership and the input-size boundary passed on macOS arm64.
The full frozen artifact matched Python on 127 inputs: IDs, byte spans, recovery
and both log probabilities. All 22 native source pipelines ran eight diagnostics
each, retaining normalization residuals.

| Native measurement | Observed value |
|---|---:|
| Warm encodes | 1,976 |
| Median / p95 / p99 | 0.074 / 0.230 / 0.347 ms |
| Original input throughput | 3.14 MB/s |
| Fresh process median | 8.44 ms |
| Sampled RSS / physical footprint | 8.62 / 5.98 MB |

Warm timing includes record-byte hashing and excludes serialization, decoding
and startup. Fresh-process timing includes artifact load, one-byte encode/decode,
JSON output, shutdown and monitoring overhead. Source checks warm caches.
These are tokenizer measurements on the recorded host, with no neural throughput
claim. Other platforms need native execution checks.

The manifest embeds the principal measurements. Full local receipts remain under
`audit/tokenizer-finalization-20261013` in the parent project: `frozen-v1`,
`frozen-v2`, `native-sources-v1`, `qualification-v2` and the native build/test logs.
The [catalogue](catalogue.md) shows how this tokenizer's observations join durable
experience without changing its frozen rows or any model parameters.
