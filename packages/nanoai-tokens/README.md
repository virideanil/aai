# nanoai-tokens

A small native tokenizer for exact bytes, explicit source spans and controllable
segmentation. It runs on its own and supplies the same implementation to nano'ai.
The C++ core evaluates a finite byte law; Python provides the user interface,
data preparation and an independent readable reference.

The included 24,664-piece model preserves arbitrary bytes and identifies its
source vocabulary. Its small authored calibration does not make it a universal
replacement for pretrained model tokenizers. Existing models retain their native
IDs, normalization and learned embedding rows.

## Install and use

Requirements: Python 3.10 or later, a C++20 compiler and OpenSSL libcrypto
development files. `pkg-config` locates OpenSSL; alternatively set
`NANOAI_OPENSSL_ROOT` to its installed prefix.

```sh
python -m pip install .
```

```python
import nanoai_tokens as tokens

codec = tokens.load()
raw = "İstanbul, 雨 and 🧬".encode() + b"\x00\xff"
record = codec.encode(raw)
assert codec.decode(record) == raw

# Choose the fewest pieces available in this exact vocabulary.
compact = codec.encode(raw, objective="minimum")
assert codec.decode(compact) == raw

# Force a finer reading at named byte offsets, retaining original detail.
fine = codec.encode(raw, boundaries=range(1, len(raw)))
assert len(fine["spans"]) == len(raw)
assert codec.decode(fine) == raw
```

All spans use half-open byte offsets. Common IDs name dictionary pieces; active
IDs name this version's model rows. Controls remain outside literal bytes. The
model and byte identities reject mixed versions and corrupt records.

```sh
nanoai-tokens inspect
nanoai-tokens encode input.bin --out tokens.json
nanoai-tokens decode tokens.json --out restored.bin
```

The CLI creates new output files. It does not overwrite an existing destination.

## Alternatives, adaptation and model data

The default chooses the most probable segmentation under the model's immutable
lexical law. The forward score sums all admissible segmentations. Minimum-piece
selection minimizes count and then chooses the most probable tied path. Explicit
boundaries restrict the admissible paths; their forward score is the original
law's mass over that subset. These numbers describe lexical paths, not meaning or
truth. Random conditional samples and expected counts expose other readings.

```python
import random
path = codec.sample(b"an observation", random.Random(7))
assert b"".join(codec.pieces[i] for i in path) == b"an observation"
counts, logp = codec.expected_counts(b"an observation")

# A new lexical version; the existing codec and model parameters stay intact.
candidate = codec.fit([b"observation and correction", b"correct the observation"])
candidate.save("domain.nmt")
```

Evaluate a fitted version on separate texts before admitting it. Lexical fitting
changes segmentation probabilities; it does not train contextual meaning. A
changed segmentation may require training the consuming model. The included
`nanoai_tokens.data` module supplies version-bound causal batches with explicit
BOS/EOS/PAD rows for a new compatible model.

## Representation and integration

Exact byte spans are the common boundary. They connect to source catalogues,
relational alternatives and learned representations without equating those
things. Binary and ternary numerical layouts, BF16 preservation, semantic
0/1/both states and memory remain typed operators at their existing owners.
Token refinement preserves the source; semantic revision carries its evidence.

The original Python implementation is available as
`nanoai_tokens.reference.MasterTokenizer` for independent comparison. Normal
`Tokenizer` calls execute the C++ core in process. Native C++ source and the
frozen model are installed with the package; `native_source()` and `frozen_path()`
locate them. Source declarations and captured license files remain beside the
artifact. Their individual terms retain their scope; this package assigns no
new blanket license to imported data.

The current platform qualification and measured costs are recorded with the
project's comparison evidence. Foreign platform execution and a general learned
quality advantage require their own measurements.
