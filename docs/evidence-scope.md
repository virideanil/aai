# Evidence supplied with the project

This private repository contains source, tests, frozen tokenizer data and
licenses, authored evaluation inputs, complete research-design snapshots,
107 requirement statements, the reviewed pi proof and its reproducible numerical
and relational checks. It contains no model weights or live learning state.

The research snapshots mark records retained on the original host as **local
evidence**. Those references preserve provenance; they are not claims that the
referenced files are available in this checkout. Raw conversations, screenshots,
model outputs, native journals, feature caches and machine-local runtime paths
remain at their original local owners. Ask for a specific relevant non-sensitive
record when source inspection would change a decision.

The [export manifest](research-context-manifest.json) binds the snapshots to their
source bytes with whole-file SHAKE256/64-byte digests. Its transformations cover
portable links, explicitly unavailable evidence, two reviewed editorial fixes to
the pi proof and omission of original message fields from the requirement export.
Every requirement statement is retained. A digest establishes source identity;
the checked consumer establishes behavior.

The [handoff](start-here.md#measured-starting-point) and [portable verification receipt](verification.json) record the local measured
baseline. The [results](results.md) retains earlier negative outcomes and the
limitations of native and learned experiments. Reproduce a changed path in your
actual environment before extending its claim. Cross-platform builds, browser
interaction, complete learned task quality and human acceptance have separate
verification requirements.

The bundled reference commands are:

```sh
python tools/verify_relational_math.py --output /tmp/relational-math.json
python tools/verify_pi_refinement.py --output /tmp/pi-refinement.json
python -m unittest discover -s tests -p 'test_relations.py' -v
```

Use an appropriate output directory on your host. The first command verifies
finite relation/refinement/extension and exact polynomial reference cases. The
second executes the shipped HTML's pure arithmetic through Node.js and compares
it with independent Python integer and rational calculations. The third exercises
the actual source, journal, correction, carrier and recovery implementation.
These checks make no model call or external action.
