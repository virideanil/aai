# Language and perception in one journal

The catalogue records what the system observed, where it came from and which
representation produced it. It uses the shared work journal, so source
revision, restart and task context have the same meaning as elsewhere in nano'ai.
It records literal evidence through explicit calls and automatically observes
artifacts produced by the writing consumer. Catalogue observation performs no
neural training or background collection.

## A complete text example

Run from the installed checkout. New states use standard-library SQLite; legacy
SJQL states retain their optional source dependency. The tokenizer file ships in
this checkout. Choose a new state directory for the first run.

```python
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.catalogue import LanguageCatalogue
from nanoai.master_tokenizer import MasterTokenizer
from nanoai.memory import Memory

codec = MasterTokenizer.load("data/master-tokenizer-v1/master.nmt")
memory = Memory(".local/catalogue-example")
memory.initialize(DEFAULT_IDENTITY)
memory.put_source("caption", "The lamp is lit. Lamba yanıyor.", note="Authored example")
memory.set_task("read", "Keep the wording and its source")
catalogue = LanguageCatalogue(memory)
catalogue.observe("reading-1", "read", "caption", codec)

view = LanguageCatalogue(Memory(memory.root)).view("read")
assert view["observations"][0]["current"]
assert sum(piece["uses"] for piece in view["pieces"]) > 0
print({"distinct_pieces": view["total_pieces"], "updates": view["parameter_updates"]})

memory.put_source("caption", "The lamp is off. Lamba sönük.", expected_revision=1)
assert catalogue.view("read")["pieces"] == []
history = catalogue.view("read", include_history=True)
assert not history["observations"][0]["current"]
catalogue.observe("reading-2", "read", "caption", codec)
assert catalogue.view("read")["pieces"]
```

A retry with the same observation ID and identical evidence is idempotent.
Reusing that ID for changed evidence is rejected. Correcting a source makes its
old observation historical; a new ID records the revised source. An unrelated
source insertion leaves the observation current.

Observation summaries retain prior readings and explain why they became
historical. The default piece counts use current readings. `--history` also
includes historical piece occurrences and perception references.

The CLI exposes the same operations:

```sh
nanoai --state .local/catalogue-example catalogue read
nanoai --state .local/catalogue-example catalogue read --history
nanoai --state .local/catalogue-example dictionary read --source caption
```

`catalogue-observe --help` lists the explicit source, tokenizer and expected
identity arguments. It adds the observation after checking the complete frozen
artifact identity. `catalogue --inventory PATH` additionally reads source witnesses
from the matching cold inventory. The inventory is prepared separately by
`tools/prepare_master_sources.py`; raw third-party payloads retain their own source
terms and are kept outside this repository.

## What each record means

| Field | Meaning |
|---|---|
| Barcode | Full SHAKE256, 64-byte output, over the exact piece bytes |
| Common ID | Source-dictionary identity and the piece's ID in that dictionary |
| Active ID | Row used by the frozen tokenizer's consuming model |
| Span | Half-open byte interval in the original UTF-8 source text |
| Native witness | Source tokenizer, exact source version, native ID and contextual role |
| Uses and followers | Counts within retained observations; follower edges stop at each observation boundary |
| Current status | The task and source revisions still match the recorded snapshot |

The barcode is `SHAKE256(piece_bytes)` with no domain prefix. Older browser
catalogue formats with a prefix have a separate identity scheme and need an
explicit migration. A partial UTF-8 piece is shown in hex and marked
`utf8_complete: false`. A literal occurrence establishes neither the word's
meaning nor a correct inference about its source.

Views return at most the requested 1–5,000 pieces and report remaining counts.
The dictionary includes up to 64 current catalogue/perception entries and states
how many remain. These are presentation bounds; the journal retains its events.
A single inventory witness view requires observations from one inventory version.

## Attach a SEE observation

Install the `perception` extra and supply an existing native SEE executable and a
sealed `see.bundle/1` database. The port opens the bundle read-only. It checks the
source sample identity through native decode/encode, then checks each requested
mask's native identity, dimensions, area and bounding box. Supplying the original
image additionally binds the recorded file bytes. For example:

```python
from nanoai.perception import attach_perception

reference = attach_perception(
    memory, "image-1", "read", "caption",
    bundle="/path/to/painting.see.sqlite",
    executable="/path/to/see",
    region_ids=[123],
    source_image="/path/to/painting.png",
)
```

Select actual region IDs from the chosen bundle. The port accepts 1–16 distinct
regions, a sealed bundle up to 256 MiB, and a source plane up to 16,777,216 pixels.
Coordinates are source pixels: x points right, y points down, and boxes are
half-open. This version records an identity transform. A rescaled or transformed
observation needs its own checked coordinate mapping.

Native sample and mask identities preserve SEE's `TupleHash256` domains and
dimensions. Engine readings and the source's attributed description are separate
from the measured samples. Current views recheck the bundle and optional original
file; changed bytes or revised source text make the reference historical.

## Checked consumer and remaining work

The actual Monet *Water Lilies* consumer checked an 1800 × 1732 native source and
two masks, recorded 31 distinct text pieces with 581 native witnesses, recovered
the same catalogue through a fresh CLI process and exposed it in the task
dictionary. Correcting its caption invalidated both language and perception
records, while a new text observation succeeded. The run made zero parameter
updates. Its original evidence remains in the parent project's
`audit/tokenizer-finalization-20261013/catalogue-consumer-v1`.

`tools/qualify_catalogue.py --help` gives the full executable recipe with explicit
artifact, inventory, bundle, source image, native engine and new output directory.
A separate synthetic 16-bit RGBA source checks exact samples, including hidden
color under zero alpha, and a mask digest against an independent TupleHash
implementation. These checks qualify source integration and recovery. The writing
consumer now observes its actual generated artifacts in the same catalogue;
useful learned semantic interpretation remains separately evaluated.
